"""Compact float16 embedding storage; no image crops are persisted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def box_key(sequence: str, frame: int, track_id: int) -> str:
    return f"{sequence}|{int(frame)}|{int(track_id)}"


class EmbeddingStoreWriter:
    def __init__(self, output_dir: str | Path, split: str, count: int, dimension: int = 512):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.count = count
        self.dimension = dimension
        self.embedding_path = self.output_dir / f"embeddings_{split}.npy"
        self.metadata_path = self.output_dir / f"metadata_{split}.jsonl"
        self.index_path = self.output_dir / f"index_{split}.json"
        self.info_path = self.output_dir / f"store_{split}.json"
        self.embeddings = np.lib.format.open_memmap(
            self.embedding_path, mode="w+", dtype=np.float16, shape=(count, dimension)
        )
        self.metadata = self.metadata_path.open("w", encoding="utf-8")
        self.index: dict[str, int] = {}
        self.next_index = 0

    def add(self, sequence: str, frame: int, track_id: int, tlwh: Iterable[float], embedding: np.ndarray) -> int:
        if self.next_index >= self.count:
            raise IndexError("embedding writer received more rows than expected")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(f"expected embedding dimension {self.dimension}, got {vector.shape[0]}")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("embedding is not finite and non-zero")
        vector = vector / norm
        key = box_key(sequence, frame, track_id)
        if key in self.index:
            raise ValueError(f"duplicate embedding key: {key}")
        row = self.next_index
        self.embeddings[row] = vector.astype(np.float16)
        self.index[key] = row
        self.metadata.write(
            json.dumps(
                {
                    "index": row,
                    "key": key,
                    "sequence": sequence,
                    "frame": int(frame),
                    "track_id": int(track_id),
                    "tlwh": [float(value) for value in tlwh],
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.next_index += 1
        return row

    def close(self, encoder_metadata: dict[str, object], extra: dict[str, object] | None = None) -> None:
        if self.next_index != self.count:
            raise ValueError(f"embedding writer expected {self.count} rows but wrote {self.next_index}")
        self.embeddings.flush()
        del self.embeddings
        self.metadata.close()
        self.index_path.write_text(json.dumps(self.index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        info = {
            "format": "N72R16-float16-npy-v1",
            "split": self.split,
            "rows": self.count,
            "dimension": self.dimension,
            "dtype": "float16",
            "encoder": encoder_metadata,
            "images_or_crops_saved": False,
        }
        if extra:
            info.update(extra)
        self.info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class EmbeddingStore:
    def __init__(self, output_dir: str | Path, split: str, mmap: bool = True):
        root = Path(output_dir)
        self.split = split
        self.info = json.loads((root / f"store_{split}.json").read_text(encoding="utf-8"))
        self.index: dict[str, int] = json.loads((root / f"index_{split}.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(root / f"embeddings_{split}.npy", mmap_mode="r" if mmap else None)
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != int(self.info["dimension"]):
            raise ValueError("embedding store shape does not match its metadata")

    def get(self, sequence: str, frame: int, track_id: int) -> np.ndarray:
        key = box_key(sequence, frame, track_id)
        try:
            row = self.index[key]
        except KeyError as exc:
            raise KeyError(f"embedding missing for {key}") from exc
        return np.asarray(self.embeddings[row], dtype=np.float32)

    def has(self, sequence: str, frame: int, track_id: int) -> bool:
        return box_key(sequence, frame, track_id) in self.index
