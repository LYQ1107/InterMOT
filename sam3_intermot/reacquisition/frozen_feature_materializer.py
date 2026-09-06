"""Short-lived frozen OSNet feature materialization for N72R11R1.

The materializer intentionally owns no encoder between calls.  A caller first
finishes and releases the official SAM3 session, then uses this adapter to
materialize machine ROI features from the original frame pixels.  It accepts
only frame paths and boxes; it has no GT, public-ID or posthoc scoring input.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_DIM = 512


class FrozenOSNetFeatureMaterializer:
    """Create, use and release one frozen OSNet encoder per materialization."""

    def __init__(
        self,
        *,
        device: str,
        frame_paths: Sequence[str | Path] | Mapping[int, str | Path],
    ) -> None:
        self.device = str(device)
        self.frame_paths = frame_paths

    def _frame_path(self, frame: int) -> Path:
        index = int(frame)
        try:
            raw = self.frame_paths[index]  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            raise ValueError(f"feature frame path is unavailable for global frame {index}") from None
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"feature frame path is not a regular file: {path}")
        return path

    @staticmethod
    def _feature_hash(value: np.ndarray) -> str:
        return hashlib.sha256(
            np.asarray(value, dtype="<f4").reshape(-1).tobytes()
        ).hexdigest()

    @staticmethod
    def _release_encoder(encoder: Any | None) -> None:
        del encoder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _new_encoder(self) -> Any:
        # Lazy import keeps this module usable for CPU-only schema checks and
        # ensures no encoder is constructed by __init__.
        from scripts.n72r5_stage07_official_full_loop import FrozenMachineOSNetN72R5

        return FrozenMachineOSNetN72R5(self.device)

    @staticmethod
    def _check_features(values: Any, expected_count: int) -> np.ndarray:
        features = np.asarray(values, dtype=np.float32).reshape(expected_count, -1)
        if features.shape != (expected_count, FEATURE_DIM) or not np.all(np.isfinite(features)):
            raise RuntimeError(
                f"frozen machine ROI feature has invalid shape/values: {features.shape}"
            )
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms <= 1.0e-6):
            raise RuntimeError("frozen machine ROI feature has zero norm")
        return features / norms[:, None]

    def materialize_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_source: str,
    ) -> list[dict[str, Any]]:
        """Add frozen 512-D features to geometry rows after SAM3 is closed."""

        if not str(feature_source):
            raise ValueError("feature_source must be non-empty")
        result = [deepcopy(dict(row)) for row in rows]
        by_frame: dict[int, list[int]] = {}
        for index, row in enumerate(result):
            frame = int(row.get("frame", -1))
            if frame < 0:
                raise ValueError("feature row lacks a non-negative global frame")
            box = np.asarray(row.get("box_xyxy"), dtype=np.float64).reshape(-1)
            if box.size != 4 or not np.all(np.isfinite(box)):
                raise ValueError(f"feature row has invalid box at frame {frame}")
            by_frame.setdefault(frame, []).append(index)

        for frame in sorted(by_frame):
            indices = by_frame[frame]
            boxes = [result[index]["box_xyxy"] for index in indices]
            encoder: Any | None = None
            try:
                encoder = self._new_encoder()
                features = self._check_features(
                    encoder.encode(self._frame_path(frame), boxes), len(indices)
                )
                for row_index, feature in zip(indices, features):
                    row = result[row_index]
                    row["feature"] = feature.astype(float).tolist()
                    row["feature_dim"] = FEATURE_DIM
                    row["feature_sha256"] = self._feature_hash(feature)
                    row["feature_source"] = str(feature_source)
            finally:
                self._release_encoder(encoder)
                encoder = None
        return result

    def materialize_human_anchor(
        self,
        *,
        frame: int,
        box_xyxy: Sequence[float],
    ) -> np.ndarray:
        """Materialize one current-frame machine ROI anchor and release OSNet."""

        box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("human anchor box must be a finite positive XYXY box")
        encoder: Any | None = None
        try:
            encoder = self._new_encoder()
            values = self._check_features(
                encoder.encode(self._frame_path(int(frame)), [box.tolist()]), 1
            )
            return values[0].copy()
        finally:
            self._release_encoder(encoder)
            encoder = None


__all__ = ["FEATURE_DIM", "FrozenOSNetFeatureMaterializer"]
