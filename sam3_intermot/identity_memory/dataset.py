"""Batched access to the frozen N72R17 protocol and embedding stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Collection

import numpy as np
import torch

from sam3_intermot.identity_research.protocol import Anchor, iter_anchors, read_frozen_protocol
from sam3_intermot.identity_research.storage import EmbeddingStore

from .encoder import FEATURE_DIMENSION


@dataclass(frozen=True)
class EpisodeMeta:
    """Metadata retained for record writing; embeddings remain in the store."""

    anchor: Anchor


class FrozenIdentityEpisodeDataset:
    """Read-only episodes built from an inherited N72R17 protocol.

    Each batch is padded only in memory.  The dataset never writes crops or
    changes the protocol.  Every future target feature is available as an
    offline same-identity ground-truth observation for the causal replay.
    """

    def __init__(
        self,
        protocol_path: str | Path,
        store_root: str | Path,
        split: str,
        sequence_filter: Collection[str] | None = None,
    ):
        self.protocol_path = str(Path(protocol_path).resolve())
        self.store_root = str(Path(store_root).resolve())
        self.split = split
        self.document = read_frozen_protocol(self.protocol_path)
        anchors = tuple(iter_anchors(self.document))
        if sequence_filter is not None:
            allowed = frozenset(str(item) for item in sequence_filter)
            anchors = tuple(anchor for anchor in anchors if anchor.sequence in allowed)
        self.sequence_filter = None if sequence_filter is None else frozenset(str(item) for item in sequence_filter)
        self.anchors = anchors
        self.store = EmbeddingStore(self.store_root, split=split)
        if int(self.store.info.get("dimension", -1)) != FEATURE_DIMENSION:
            raise ValueError("N72R18 requires the inherited 512-D OSNet embedding store")
        if not self.anchors:
            raise ValueError(f"protocol contains no anchors: {self.protocol_path}")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> EpisodeMeta:
        return EpisodeMeta(self.anchors[index])

    def batch_indices(self, batch_size: int, shuffle: bool = False, seed: int = 7218) -> Iterable[list[int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [int(index) for index in indices[start : start + batch_size]]

    def make_batch(self, indices: list[int], device: torch.device | str | None = None) -> dict[str, object]:
        if not indices:
            raise ValueError("cannot build an empty batch")
        anchors = [self.anchors[index] for index in indices]
        max_steps = max(len(anchor.future) for anchor in anchors)
        max_competitors = max(
            (len(future.competitors) for anchor in anchors for future in anchor.future),
            default=0,
        )
        batch_size = len(anchors)
        anchor_embeddings = np.zeros((batch_size, FEATURE_DIMENSION), dtype=np.float32)
        target_embeddings = np.zeros((batch_size, max_steps, FEATURE_DIMENSION), dtype=np.float32)
        target_mask = np.zeros((batch_size, max_steps), dtype=bool)
        competitor_embeddings = np.zeros(
            (batch_size, max_steps, max_competitors, FEATURE_DIMENSION), dtype=np.float32
        )
        competitor_mask = np.zeros((batch_size, max_steps, max_competitors), dtype=bool)
        frames = np.zeros((batch_size, max_steps), dtype=np.int64)
        gaps = np.zeros((batch_size, max_steps), dtype=np.int64)

        for batch_index, anchor in enumerate(anchors):
            anchor_embeddings[batch_index] = self.store.get(anchor.sequence, anchor.anchor_frame, anchor.track_id)
            for step_index, future in enumerate(anchor.future):
                target_embeddings[batch_index, step_index] = self.store.get(
                    anchor.sequence, future.frame, anchor.track_id
                )
                target_mask[batch_index, step_index] = True
                frames[batch_index, step_index] = future.frame
                gaps[batch_index, step_index] = future.frame - anchor.anchor_frame
                for competitor_index, competitor in enumerate(future.competitors):
                    competitor_embeddings[batch_index, step_index, competitor_index] = self.store.get(
                        anchor.sequence, future.frame, competitor.track_id
                    )
                    competitor_mask[batch_index, step_index, competitor_index] = True

        result: dict[str, object] = {
            "anchor_embeddings": torch.from_numpy(anchor_embeddings),
            "target_embeddings": torch.from_numpy(target_embeddings),
            "target_mask": torch.from_numpy(target_mask),
            "competitor_embeddings": torch.from_numpy(competitor_embeddings),
            "competitor_mask": torch.from_numpy(competitor_mask),
            "frames": torch.from_numpy(frames),
            "gaps": torch.from_numpy(gaps),
            "metadata": [EpisodeMeta(anchor) for anchor in anchors],
        }
        if device is not None:
            for key, value in list(result.items()):
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(device, non_blocking=True)
        return result


__all__ = ["EpisodeMeta", "FrozenIdentityEpisodeDataset"]
