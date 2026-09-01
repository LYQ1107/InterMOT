"""Identity-scoped LIT-style LoRA for the official SAM3 propagation decoder.

The official SAM3 model remains the single frozen base decoder.  Adapter
states live outside that module and are selected by a short-lived activation
context, so a video/identity does not receive a private copy of the decoder.
Only the exact Q/K/V projections used by the LIT mask-decoder recipe are
wrapped.  The original ``nn.Linear`` (including its bias) is kept intact.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LIT_TARGET_SUFFIXES: tuple[str, ...] = (
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.cross_attn_token_to_image.q_proj",
    "layers.0.cross_attn_token_to_image.k_proj",
    "layers.0.cross_attn_token_to_image.v_proj",
    "layers.0.cross_attn_image_to_token.q_proj",
    "layers.0.cross_attn_image_to_token.k_proj",
    "layers.0.cross_attn_image_to_token.v_proj",
    "layers.1.self_attn.q_proj",
    "layers.1.self_attn.k_proj",
    "layers.1.self_attn.v_proj",
    "layers.1.cross_attn_token_to_image.q_proj",
    "layers.1.cross_attn_token_to_image.k_proj",
    "layers.1.cross_attn_token_to_image.v_proj",
    "layers.1.cross_attn_image_to_token.q_proj",
    "layers.1.cross_attn_image_to_token.k_proj",
    "layers.1.cross_attn_image_to_token.v_proj",
    "final_attn_token_to_image.q_proj",
    "final_attn_token_to_image.k_proj",
    "final_attn_token_to_image.v_proj",
)


def _safe_key(path: str) -> str:
    """Make a stable ParameterDict key without losing the source path."""

    return path.replace(".", "__")


@dataclass(frozen=True)
class ProjectionSpec:
    path: str
    in_features: int
    out_features: int

    @property
    def key(self) -> str:
        return _safe_key(self.path)

    def parameter_count(self, rank: int) -> int:
        return int(rank) * (self.in_features + self.out_features)


@dataclass(frozen=True)
class DecoderLITConfig:
    """Online LoRA settings matching the audited LIT decoder recipe."""

    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.1
    train_a: bool = True
    train_b: bool = True

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


class IdentityAdapterState(nn.Module):
    """The 21 A/B tensors and metadata for one ``(video_id, public_id)``."""

    def __init__(
        self,
        specs: Iterable[ProjectionSpec],
        config: DecoderLITConfig,
        *,
        video_id: Any,
        public_id: Any,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_id = video_id
        self.public_id = public_id
        self.adapter_version = 0
        self.correction_count = 0
        self.last_frame = -1
        self.last_provenance = "NONE"
        self._specs: Dict[str, ProjectionSpec] = {spec.path: spec for spec in specs}
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        for spec in self._specs.values():
            # A is the down projection [rank, input]; B is the up projection
            # [output, rank], which gives Delta W = B @ A.
            a = torch.empty(
                config.rank,
                spec.in_features,
                device=device,
                dtype=dtype,
            )
            b = torch.zeros(
                spec.out_features,
                config.rank,
                device=device,
                dtype=dtype,
            )
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_a[spec.key] = nn.Parameter(a, requires_grad=config.train_a)
            self.lora_b[spec.key] = nn.Parameter(b, requires_grad=config.train_b)

    @property
    def specs(self) -> Mapping[str, ProjectionSpec]:
        return self._specs

    @property
    def identity_key(self) -> str:
        return f"{self.video_id!r}\x1f{self.public_id!r}"

    def tensors(self, path: str) -> tuple[nn.Parameter, nn.Parameter]:
        spec = self._specs[path]
        return self.lora_a[spec.key], self.lora_b[spec.key]

    def parameters_for_update(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def parameter_count(self) -> int:
        return sum(
            spec.parameter_count(self.config.rank) for spec in self._specs.values()
        )

    def mark_update(self, frame: int, provenance: str) -> None:
        self.adapter_version += 1
        self.correction_count += 1
        self.last_frame = int(frame)
        self.last_provenance = str(provenance)

    def export_cpu(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "public_id": self.public_id,
            "adapter_version": self.adapter_version,
            "correction_count": self.correction_count,
            "last_frame": self.last_frame,
            "last_provenance": self.last_provenance,
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "train_a": self.config.train_a,
                "train_b": self.config.train_b,
            },
            "lora_a": {
                path: self.lora_a[spec.key].detach().cpu().clone()
                for path, spec in self._specs.items()
            },
            "lora_b": {
                path: self.lora_b[spec.key].detach().cpu().clone()
                for path, spec in self._specs.items()
            },
        }


@dataclass(frozen=True)
class AdapterSnapshot:
    """Rollback material for one identity adapter."""

    tensors: Mapping[str, tuple[Tensor, Tensor]]
    adapter_version: int
    correction_count: int
    last_frame: int
    last_provenance: str


class DecoderAdapterBank:
    """Per-video/per-public-ID adapter states, without model copies."""

    def __init__(self, specs: Iterable[ProjectionSpec], config: DecoderLITConfig):
        self.specs = tuple(specs)
        self.config = config
        self._states: dict[str, IdentityAdapterState] = {}

    @staticmethod
    def make_key(video_id: Any, public_id: Any) -> str:
        raw = f"{video_id!r}\x1f{public_id!r}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def get_or_create(
        self,
        video_id: Any,
        public_id: Any,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> IdentityAdapterState:
        key = self.make_key(video_id, public_id)
        state = self._states.get(key)
        if state is None:
            state = IdentityAdapterState(
                self.specs,
                self.config,
                video_id=video_id,
                public_id=public_id,
                device=device,
                dtype=dtype,
            )
            self._states[key] = state
        elif device is not None:
            state.to(device=device, dtype=dtype)
        return state

    def get(self, video_id: Any, public_id: Any) -> Optional[IdentityAdapterState]:
        return self._states.get(self.make_key(video_id, public_id))

    def __len__(self) -> int:
        return len(self._states)

    def states(self) -> tuple[IdentityAdapterState, ...]:
        return tuple(self._states.values())

    def snapshot(self, state: IdentityAdapterState) -> AdapterSnapshot:
        return AdapterSnapshot(
            tensors={
                path: (
                    state.lora_a[spec.key].detach().clone(),
                    state.lora_b[spec.key].detach().clone(),
                )
                for path, spec in state.specs.items()
            },
            adapter_version=state.adapter_version,
            correction_count=state.correction_count,
            last_frame=state.last_frame,
            last_provenance=state.last_provenance,
        )

    def restore(self, state: IdentityAdapterState, snapshot: AdapterSnapshot) -> None:
        with torch.no_grad():
            for path, (a, b) in snapshot.tensors.items():
                spec = state.specs[path]
                state.lora_a[spec.key].copy_(a)
                state.lora_b[spec.key].copy_(b)
        state.adapter_version = snapshot.adapter_version
        state.correction_count = snapshot.correction_count
        state.last_frame = snapshot.last_frame
        state.last_provenance = snapshot.last_provenance


class IdentityScopedLoRALinear(nn.Module):
    """A transparent wrapper whose active A/B tensors come from a bank."""

    def __init__(self, base: nn.Linear, config: DecoderLITConfig):
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scale = float(config.alpha) / float(config.rank)
        self.dropout = float(config.dropout)
        self._active_a: Optional[Tensor] = None
        self._active_b: Optional[Tensor] = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.base.bias

    def set_active(self, a: Optional[Tensor], b: Optional[Tensor]) -> None:
        if (a is None) != (b is None):
            raise ValueError("LoRA A and B must be activated together")
        self._active_a = a
        self._active_b = b

    def clear_active(self) -> None:
        self._active_a = None
        self._active_b = None

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        if self._active_a is None or self._active_b is None:
            self._debug_last_forward = {
                "active": False,
                "grad_enabled": bool(torch.is_grad_enabled()),
                "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
                "base_requires_grad": bool(base_out.requires_grad),
                "delta_requires_grad": False,
                "output_requires_grad": bool(base_out.requires_grad),
            }
            return base_out
        a = self._active_a.to(device=x.device, dtype=x.dtype)
        b = self._active_b.to(device=x.device, dtype=x.dtype)
        dropped = F.dropout(x, p=self.dropout, training=self.training)
        # Use the explicit batched-matmul form of B @ (A @ x).  On the
        # torch-2.5/CUDA runtime used by the pinned SAM3 environment,
        # ``F.linear`` can select an inference kernel for this frozen-base /
        # external-parameter combination even with grad mode enabled.  The
        # equivalent matmul keeps the LoRA graph differentiable without
        # changing its numerical definition.
        delta = torch.matmul(
            torch.matmul(dropped, a.transpose(-1, -2)), b.transpose(-1, -2)
        )
        output = base_out + delta * self.scale
        self._debug_last_forward = {
            "active": True,
            "grad_enabled": bool(torch.is_grad_enabled()),
            "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
            "base_requires_grad": bool(base_out.requires_grad),
            "delta_requires_grad": bool(delta.requires_grad),
            "output_requires_grad": bool(output.requires_grad),
            "a_requires_grad": bool(a.requires_grad),
            "b_requires_grad": bool(b.requires_grad),
            "input_is_inference": bool(torch.is_inference(x)),
            "a_is_inference": bool(torch.is_inference(a)),
            "b_is_inference": bool(torch.is_inference(b)),
            "dropped_requires_grad": bool(dropped.requires_grad),
            "dropped_is_inference": bool(torch.is_inference(dropped)),
            "a_dtype": str(a.dtype),
            "b_dtype": str(b.dtype),
            "input_dtype": str(x.dtype),
        }
        return output


class SAM3DecoderLITAdapter:
    """Inject and activate exact LIT targets on one official decoder."""

    def __init__(
        self,
        decoder_or_transformer: nn.Module,
        config: Optional[DecoderLITConfig] = None,
    ) -> None:
        self.config = config or DecoderLITConfig()
        self.base_module = decoder_or_transformer
        self.transformer = getattr(
            decoder_or_transformer, "transformer", decoder_or_transformer
        )
        if not isinstance(self.transformer, nn.Module):
            raise TypeError("decoder_or_transformer must expose an nn.Module transformer")
        self.wrappers: dict[str, IdentityScopedLoRALinear] = {}
        specs: list[ProjectionSpec] = []
        for path, module in list(self.transformer.named_modules()):
            if path not in LIT_TARGET_SUFFIXES:
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(f"official LoRA target is not Linear: {path}")
            parent_name, _, child_name = path.rpartition(".")
            parent = self.transformer if not parent_name else self._get_module(parent_name)
            wrapped = IdentityScopedLoRALinear(module, self.config)
            wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, child_name, wrapped)
            specs.append(ProjectionSpec(path, module.in_features, module.out_features))
            self.wrappers[path] = wrapped

        expected = set(LIT_TARGET_SUFFIXES)
        actual = set(self.wrappers)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(f"exact 21-target audit failed; missing={missing}, extra={extra}")
        # The complete official decoder is an inference anchor.  Freeze its
        # transformer and mask-head parameters; adapter parameters live in the
        # external identity bank and therefore are not touched by this loop.
        # Freezing the non-transformer head is also important for replaying
        # inputs captured under the official inference-mode executor.
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        self.specs = tuple(specs)
        self.bank = DecoderAdapterBank(self.specs, self.config)

    def _get_module(self, path: str) -> nn.Module:
        node: nn.Module = self.transformer
        for part in path.split("."):
            node = getattr(node, part)
        return node

    @property
    def target_paths(self) -> tuple[str, ...]:
        return tuple(spec.path for spec in self.specs)

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def new_state(
        self,
        video_id: Any,
        public_id: Any,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> IdentityAdapterState:
        return self.bank.get_or_create(
            video_id,
            public_id,
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    @contextlib.contextmanager
    def activate(
        self, state: Optional[IdentityAdapterState]
    ) -> Iterator[Optional[IdentityAdapterState]]:
        """Activate one identity for one official decoder call."""

        if state is not None:
            if set(state.specs) != set(self.target_paths):
                raise ValueError("adapter state target inventory does not match decoder")
            for path, wrapper in self.wrappers.items():
                spec = state.specs[path]
                wrapper.set_active(
                    state.lora_a[spec.key],
                    state.lora_b[spec.key],
                )
        try:
            yield state
        finally:
            for wrapper in self.wrappers.values():
                wrapper.clear_active()

    def snapshot(self, state: IdentityAdapterState) -> AdapterSnapshot:
        return self.bank.snapshot(state)

    def restore(self, state: IdentityAdapterState, snapshot: AdapterSnapshot) -> None:
        self.bank.restore(state, snapshot)

    def inventory(self, state: Optional[IdentityAdapterState] = None) -> dict[str, Any]:
        """Return auditable target/trainability/gradient inventory."""

        trainable = []
        if state is not None:
            for path, spec in state.specs.items():
                a, b = state.tensors(path)
                trainable.extend(
                    [
                        {
                            "name": f"{path}.lora_a",
                            "shape": list(a.shape),
                            "requires_grad": bool(a.requires_grad),
                            "grad_nonzero": bool(
                                a.grad is not None and torch.any(a.grad != 0).item()
                            ),
                            "parameters": int(a.numel()),
                        },
                        {
                            "name": f"{path}.lora_b",
                            "shape": list(b.shape),
                            "requires_grad": bool(b.requires_grad),
                            "grad_nonzero": bool(
                                b.grad is not None and torch.any(b.grad != 0).item()
                            ),
                            "parameters": int(b.numel()),
                        },
                    ]
                )
        return {
            "target_count": len(self.specs),
            "target_paths": list(self.target_paths),
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "adapter_parameter_count": (
                sum(spec.parameter_count(self.config.rank) for spec in self.specs)
                if state is None
                else state.parameter_count()
            ),
            "trainable": trainable,
            "base_trainable_parameters": sum(
                int(parameter.numel())
                for parameter in self.base_module.parameters()
                if parameter.requires_grad
            ),
        }


def inject_propagation_decoder_lora(
    decoder: nn.Module,
    config: Optional[DecoderLITConfig] = None,
) -> SAM3DecoderLITAdapter:
    """Explicitly named entry point used by N29 scripts and integrations."""

    return SAM3DecoderLITAdapter(decoder, config=config)


def zero_lora_state(adapter: SAM3DecoderLITAdapter, state: IdentityAdapterState) -> None:
    """Reset B to zero while retaining the Kaiming-initialized A tensors."""

    with torch.no_grad():
        for path in adapter.target_paths:
            _, b = state.tensors(path)
            b.zero_()
