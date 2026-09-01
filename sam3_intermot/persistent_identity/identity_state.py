"""Persistent per-identity state S_i (working definition)."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


LIFECYCLE = ("ACTIVE", "UNCERTAIN", "DORMANT", "REACTIVATED", "TERMINATED")


@dataclass
class PersistentIdentityState:
    """S_i = {H_i, Q_i^det, P_i^trk, M_i, B_i, L_i, C_i, ID_i}.

    Only Q_i^det is required for F0 (frozen-decoder) experiments.  P_i^trk and
    M_i are reserved for the detector-tracker bridge later.
    """

    public_id: int
    source: str = "HUMAN_SEEDED"
    slot: Optional[int] = None
    human_box: Optional[np.ndarray] = None  # authoritative anchor H_i
    query: Optional[torch.Tensor] = None  # Q_i^det
    tracker_pointer: Optional[object] = None  # P_i^trk (future bridge)
    machine_box: Optional[np.ndarray] = None  # B_i / M_i
    lifecycle: str = "ACTIVE"
    confidence: float = 0.0
    age: int = 0
    last_observed_frame: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "public_id": self.public_id,
            "source": self.source,
            "slot": self.slot,
            "human_box": None if self.human_box is None else self.human_box.tolist(),
            "lifecycle": self.lifecycle,
            "confidence": self.confidence,
            "age": self.age,
            "last_observed_frame": self.last_observed_frame,
            "has_query": self.query is not None,
        }
