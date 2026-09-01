"""Atomic, future-only storage and injection for N30 correction residuals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CorrectionMemoryState:
    video_id: str
    public_id: int
    correction_frame: int
    correction_version: int
    residual: Tensor
    gate: Tensor


class CorrectionMemoryTransaction:
    """Latest-correction replacement with explicit future-only enforcement."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, int], CorrectionMemoryState] = {}
        self.ledger: list[dict[str, Any]] = []

    def snapshot(self) -> dict[tuple[str, int], CorrectionMemoryState]:
        return {
            key: CorrectionMemoryState(
                value.video_id,
                value.public_id,
                value.correction_frame,
                value.correction_version,
                value.residual.detach().clone(),
                value.gate.detach().clone(),
            )
            for key, value in self._latest.items()
        }

    def restore(self, snapshot: dict[tuple[str, int], CorrectionMemoryState]) -> None:
        self._latest = {
            key: CorrectionMemoryState(
                value.video_id,
                value.public_id,
                value.correction_frame,
                value.correction_version,
                value.residual.detach().clone(),
                value.gate.detach().clone(),
            )
            for key, value in snapshot.items()
        }

    def write_latest(
        self,
        *,
        video_id: str,
        public_id: int,
        correction_frame: int,
        residual: Tensor,
        gate: Tensor,
    ) -> CorrectionMemoryState:
        key = (str(video_id), int(public_id))
        previous = self._latest.get(key)
        version = 1 if previous is None else previous.correction_version + 1
        state = CorrectionMemoryState(
            video_id=str(video_id),
            public_id=int(public_id),
            correction_frame=int(correction_frame),
            correction_version=version,
            residual=residual.detach().clone(),
            gate=gate.detach().clone(),
        )
        self._latest[key] = state
        self.ledger.append(
            {
                "operation": "latest_replace",
                "video_id": state.video_id,
                "public_id": state.public_id,
                "correction_frame": state.correction_frame,
                "correction_version": state.correction_version,
                "future_only": True,
            }
        )
        return state

    def get_latest(self, video_id: str, public_id: int) -> Optional[CorrectionMemoryState]:
        return self._latest.get((str(video_id), int(public_id)))

    @staticmethod
    def apply_to_extra(
        extra_per_object_embeddings: Tensor,
        *,
        state: CorrectionMemoryState,
        target_slot: int,
        frame_idx: int,
    ) -> Tensor:
        """Apply only after correction and only to the mapped target slot."""

        if int(frame_idx) <= int(state.correction_frame):
            return extra_per_object_embeddings
        extra = extra_per_object_embeddings
        if extra.ndim != 3 or extra.shape[-1] != 256:
            raise ValueError(f"expected [B,N,256] extra embeddings, got {tuple(extra.shape)}")
        if not 0 <= int(target_slot) < extra.shape[1]:
            raise IndexError(f"target slot {target_slot} outside {extra.shape[1]} object tokens")
        residual = state.residual.to(device=extra.device, dtype=extra.dtype)
        if residual.ndim == 2:
            residual = residual.unsqueeze(0)
        updated = extra.clone()
        updated[:, int(target_slot), :] = updated[:, int(target_slot), :] + residual[:, int(target_slot), :]
        return updated

    def summary(self) -> dict[str, Any]:
        return {
            "state_count": len(self._latest),
            "keys": [
                {
                    "video_id": state.video_id,
                    "public_id": state.public_id,
                    "correction_frame": state.correction_frame,
                    "correction_version": state.correction_version,
                }
                for state in self._latest.values()
            ],
            "latest_replacement": True,
            "future_only": True,
        }


class FutureExtraEmbeddingHook:
    """Project-level decoder hook used by real future inference tests."""

    def __init__(
        self,
        decoder: nn.Module,
        transaction: CorrectionMemoryTransaction,
        *,
        state: CorrectionMemoryState,
        target_slot: int,
        frame_provider: Callable[[], int],
    ) -> None:
        self.transaction = transaction
        self.state = state
        self.target_slot = int(target_slot)
        self.frame_provider = frame_provider
        self.calls = 0
        self.modified_calls = 0
        self.modified_slots: list[int] = []
        self.handle = decoder.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.calls += 1
        frame = int(self.frame_provider())
        extra = kwargs.get("extra_per_object_embeddings")
        if extra is None or frame <= self.state.correction_frame:
            return None
        modified = self.transaction.apply_to_extra(
            extra,
            state=self.state,
            target_slot=self.target_slot,
            frame_idx=frame,
        )
        updated_kwargs = dict(kwargs)
        updated_kwargs["extra_per_object_embeddings"] = modified
        self.modified_calls += 1
        self.modified_slots.append(self.target_slot)
        return args, updated_kwargs

    def close(self) -> None:
        self.handle.remove()

    def summary(self) -> dict[str, Any]:
        return {
            "decoder_calls": self.calls,
            "modified_future_calls": self.modified_calls,
            "target_slots": list(self.modified_slots),
            "future_only": True,
        }

