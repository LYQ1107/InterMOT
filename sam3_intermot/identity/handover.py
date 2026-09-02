"""Persistent lineage handover across independent SAM sessions/windows.

Handover is intentionally based on overlap observations and persistent
lineage, never on equality of raw SAM IDs from two sessions.  The matching
helper is offline with respect to GT: it consumes only current/overlap
observations and past public-track state and is suitable for a runtime
transaction builder.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

import numpy as np

from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.tracking.track_manager import TrackManager


@dataclass(frozen=True)
class HandoverTransaction:
    from_session: str
    to_session: str
    from_segment: str
    to_segment: str
    old_raw_sam_id: int
    new_raw_sam_id: int
    old_adapter_id: int
    new_adapter_id: int
    lineage_id: int
    mot_track_id: int
    public_id: int
    frame_boundary: int
    source_run_id: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def box_iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in np.asarray(a).reshape(-1)[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in np.asarray(b).reshape(-1)[:4]]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cosine(a: Any, b: Any) -> Optional[float]:
    try:
        x = np.asarray(a, dtype=np.float64).reshape(-1)
        y = np.asarray(b, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if x.size == 0 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx <= 0 or ny <= 0:
        return None
    return float(np.dot(x, y) / (nx * ny))


class PersistentLineageHandover:
    """Build and audit explicit cross-session handover transactions."""

    def __init__(self, source_run_id: str, sequence: str) -> None:
        self.source_run_id = str(source_run_id)
        self.sequence = str(sequence)
        self.transactions: list[HandoverTransaction] = []

    def add(self, transaction: HandoverTransaction) -> HandoverTransaction:
        if transaction.source_run_id != self.source_run_id:
            raise ValueError("handover source_run_id mismatch")
        if transaction.from_session == transaction.to_session:
            raise ValueError("handover requires distinct sessions")
        if transaction.from_segment == transaction.to_segment:
            raise ValueError("handover requires distinct segment identifiers")
        if transaction.status not in {"PASS", "AMBIGUOUS", "NO_MATCH", "COLLISION"}:
            raise ValueError(f"unsupported handover status: {transaction.status}")
        self.transactions.append(transaction)
        return transaction

    def match_overlap(
        self,
        previous_rows: Iterable[dict[str, Any]],
        next_rows: Iterable[dict[str, Any]],
        *,
        from_session: str,
        to_session: str,
        from_segment: str,
        to_segment: str,
        frame_boundary: int,
        min_iou: float = 0.20,
        min_score: float = 0.20,
    ) -> list[HandoverTransaction]:
        """Match new-session tracks to old lineages using overlap observations.

        Rows need ``frame_idx``, ``box``, raw/adapter IDs, and old/new public
        metadata.  A feature is optional; when available it contributes to a
        deterministic IoU/cosine score.  Each old and new track can be used
        once, with ties broken by stable IDs.  No GT field is read.
        """

        old = [dict(row) for row in previous_rows]
        new = [dict(row) for row in next_rows]
        old_by_track: dict[int, list[dict[str, Any]]] = {}
        new_by_track: dict[int, list[dict[str, Any]]] = {}
        for row in old:
            if row.get("mot_track_id") is not None:
                old_by_track.setdefault(int(row["mot_track_id"]), []).append(row)
        for row in new:
            if row.get("mot_track_id") is not None:
                new_by_track.setdefault(int(row["mot_track_id"]), []).append(row)
        pair_scores: list[tuple[float, int, int, dict[str, Any], dict[str, Any]]] = []
        for old_tid, old_track in old_by_track.items():
            for new_tid, new_track in new_by_track.items():
                best: Optional[tuple[float, dict[str, Any], dict[str, Any]]] = None
                for left in old_track:
                    for right in new_track:
                        iou = box_iou(left.get("box", [0, 0, 0, 0]), right.get("box", [0, 0, 0, 0]))
                        sim = cosine(left.get("feature", []), right.get("feature", []))
                        appearance = 0.0 if sim is None else max(0.0, sim)
                        score = 0.75 * iou + 0.25 * appearance
                        candidate = (score, left, right)
                        if best is None or (score, -int(right.get("frame_idx", 0))) > (best[0], -int(best[2].get("frame_idx", 0))):
                            best = candidate
                if best is not None and best[0] >= min_score and box_iou(best[1].get("box", [0] * 4), best[2].get("box", [0] * 4)) >= min_iou:
                    pair_scores.append((best[0], old_tid, new_tid, best[1], best[2]))
        pair_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_old: set[int] = set()
        used_new: set[int] = set()
        transactions: list[HandoverTransaction] = []
        for score, old_tid, new_tid, left, right in pair_scores:
            if old_tid in used_old or new_tid in used_new:
                continue
            old_public = left.get("public_id")
            lineage = left.get("lineage_id")
            if old_public is None or lineage is None:
                continue
            used_old.add(old_tid)
            used_new.add(new_tid)
            transaction = HandoverTransaction(
                from_session=str(from_session),
                to_session=str(to_session),
                from_segment=str(from_segment),
                to_segment=str(to_segment),
                old_raw_sam_id=int(left.get("raw_sam_id", left.get("official_raw_sam_id", -1))),
                new_raw_sam_id=int(right.get("raw_sam_id", right.get("official_raw_sam_id", -1))),
                old_adapter_id=int(left.get("adapter_id", left.get("adapter_external_id", -1))),
                new_adapter_id=int(right.get("adapter_id", right.get("adapter_external_id", -1))),
                lineage_id=int(lineage),
                mot_track_id=int(right.get("mot_track_id", new_tid)),
                public_id=int(old_public),
                frame_boundary=int(frame_boundary),
                source_run_id=self.source_run_id,
                status="PASS",
            )
            self.add(transaction)
            transactions.append(transaction)
        return transactions

    def audit(self, expected_pairs: Optional[int] = None) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for item in self.transactions:
            statuses[item.status] = statuses.get(item.status, 0) + 1
        public_ids = [item.public_id for item in self.transactions if item.status == "PASS"]
        return {
            "schema_version": "N72R2_HANDOVER_LEDGER_V1",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "transaction_count": len(self.transactions),
            "status_counts": statuses,
            "expected_pairs": expected_pairs,
            "unique_public_ids": len(set(public_ids)),
            "public_id_collisions": len(public_ids) - len(set(public_ids)),
            "raw_id_equality_used_for_match": False,
            "runtime_future_gt_used": False,
            "transactions": [item.as_dict() for item in self.transactions],
        }


def find_handover(
    manager: TrackManager,
    lineages: IdentityLineageRegistry,
    observation: Any,
    frame_idx: int,
    max_gap: int = 45,
) -> Optional[int]:
    """Backward-compatible past-state lookup used by the legacy Add path."""

    return lineages.find_lost_lineage(
        observation=observation,
        frame_idx=frame_idx,
        manager=manager,
        max_gap=max_gap,
    )


__all__ = [
    "HandoverTransaction",
    "PersistentLineageHandover",
    "box_iou",
    "cosine",
    "find_handover",
]
