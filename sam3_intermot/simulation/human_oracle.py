"""Causal GT-simulated human oracle for offline N72R3 experiments.

This module is deliberately outside ``sam3_intermot.association`` and
``sam3_intermot.identity``.  It may be used by simulation/evaluation workers
after the current-frame prediction has been frozen.  It owns a private
dataset-GT-ID -> public-ID knowledge map, but never sends dataset IDs to the
candidate, association, mapping, prompt, or memory code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np


ACTION_TYPES = (
    "AUTHORITATIVE_CORRECT",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "ADD_NEW_IDENTITY",
    "RECOVER_IDENTITY",
    "AUTHORITATIVE_DELETE",
)


def _box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)[:4]
    b = np.asarray(right, dtype=float).reshape(-1)[:4]
    if a.size != 4 or b.size != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _current_gt_items(current_gt: Any) -> list[dict[str, Any]]:
    if isinstance(current_gt, dict):
        if any(str(key).lower().startswith("future") for key in current_gt):
            raise ValueError("SimulatedHumanOracle accepts current-frame GT only")
        boxes = current_gt.get("boxes", [])
        ids = current_gt.get("gt_ids", current_gt.get("ids", []))
    else:
        boxes = getattr(current_gt, "boxes", [])
        ids = getattr(current_gt, "gt_ids", [])
    if boxes is None:
        boxes = []
    if ids is None:
        ids = []
    if len(boxes) != len(ids):
        raise ValueError("current GT boxes and IDs have different lengths")
    return [
        {"dataset_gt_id": int(gt_id), "box": np.asarray(box, dtype=float).reshape(-1)[:4]}
        for gt_id, box in zip(ids, boxes)
    ]


@dataclass(frozen=True)
class OracleDecision:
    frame_idx: int
    action_type: str
    dataset_gt_id: Optional[int]
    target_public_id: Optional[int]
    current_box: Optional[list[float]]
    matched_runtime_public_id: Optional[int]
    overlap_iou: Optional[float]
    other_public_id: Optional[int] = None
    mapping_confirmation_required: bool = False
    interaction_source: str = "simulated_from_gt"
    runtime_future_gt_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "action_type": self.action_type,
            "dataset_gt_id": self.dataset_gt_id,
            "target_public_id": self.target_public_id,
            "current_box": self.current_box,
            "matched_runtime_public_id": self.matched_runtime_public_id,
            "overlap_iou": self.overlap_iou,
            "other_public_id": self.other_public_id,
            "mapping_confirmation_required": self.mapping_confirmation_required,
            "interaction_source": self.interaction_source,
            "runtime_future_gt_used": False,
        }


@dataclass
class OracleAudit:
    current_frame_reads: int = 0
    future_frame_reads: int = 0
    runtime_future_gt_used: bool = False
    mapping_commit_count: int = 0
    event_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_frame_reads": int(self.current_frame_reads),
            "future_frame_reads": int(self.future_frame_reads),
            "runtime_future_gt_used": bool(self.future_frame_reads > 0),
            "mapping_commit_count": int(self.mapping_commit_count),
            "event_count": int(self.event_count),
        }


class SimulatedHumanOracle:
    """Choose current-frame simulated-human actions with an isolated GT map."""

    def __init__(self, sequence: str, *, known_gt_to_public: Optional[dict[int, int]] = None) -> None:
        if not sequence:
            raise ValueError("sequence is required")
        self.sequence = str(sequence)
        self._gt_to_public: dict[int, int] = {
            int(gt_id): int(public_id) for gt_id, public_id in (known_gt_to_public or {}).items()
        }
        self.audit = OracleAudit()
        self._events: list[OracleDecision] = []

    @property
    def gt_to_public(self) -> dict[int, int]:
        return dict(self._gt_to_public)

    @property
    def events(self) -> tuple[OracleDecision, ...]:
        return tuple(self._events)

    def commit_mapping(self, dataset_gt_id: int, public_id: int, *, reason: str) -> None:
        """Commit an explicit current-frame/outer-allocation mapping.

        The method is called by the simulation harness after the outer
        runtime either confirms an existing public ID or allocates a new one.
        It never derives a public ID from a candidate/native ID.
        """

        if reason not in {"current_frame_runtime_confirmation", "outer_allocator_birth"}:
            raise ValueError("mapping commit requires an explicit causal reason")
        gt_id = int(dataset_gt_id)
        public = int(public_id)
        existing = self._gt_to_public.get(gt_id)
        if existing is not None and existing != public:
            raise ValueError(f"dataset GT identity {gt_id} already maps to public ID {existing}")
        occupied = [key for key, value in self._gt_to_public.items() if value == public and key != gt_id]
        if occupied:
            raise ValueError(f"public ID {public} is already mapped to GT identities {occupied}")
        self._gt_to_public[gt_id] = public
        self.audit.mapping_commit_count += 1

    @staticmethod
    def _predictions(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, value in enumerate(predictions):
            row = dict(value)
            uid = str(row.get("candidate_uid", f"prediction-{index}"))
            if uid in seen:
                raise ValueError(f"duplicate prediction candidate UID: {uid}")
            seen.add(uid)
            if row.get("public_id") is not None:
                row["public_id"] = int(row["public_id"])
            box = np.asarray(row.get("box", [0, 0, 0, 0]), dtype=float).reshape(-1)
            if box.size != 4 or not np.isfinite(box).all():
                raise ValueError(f"invalid current prediction box: {uid}")
            row["candidate_uid"] = uid
            row["box"] = box
            result.append(row)
        return result

    def choose_actions(
        self,
        frame_idx: int,
        current_gt: Any,
        predictions: Iterable[dict[str, Any]],
        *,
        localization_iou_threshold: float = 0.5,
    ) -> list[OracleDecision]:
        """Select actions using only one current GT frame and current output."""

        if not 0.0 < float(localization_iou_threshold) <= 1.0:
            raise ValueError("localization_iou_threshold must be in (0, 1]")
        frame = int(frame_idx)
        gt_items = _current_gt_items(current_gt)
        pred_items = self._predictions(predictions)
        self.audit.current_frame_reads += 1
        self.audit.runtime_future_gt_used = bool(self.audit.future_frame_reads > 0)

        overlaps = np.zeros((len(gt_items), len(pred_items)), dtype=float)
        for gi, gt in enumerate(gt_items):
            for pi, pred in enumerate(pred_items):
                overlaps[gi, pi] = _box_iou(gt["box"], pred["box"])
        matched: dict[int, int] = {}
        used_predictions: set[int] = set()
        pairs = [
            (float(overlaps[gi, pi]), gi, pi)
            for gi in range(len(gt_items))
            for pi in range(len(pred_items))
            if overlaps[gi, pi] >= float(localization_iou_threshold)
        ]
        for _, gi, pi in sorted(pairs, key=lambda item: (-item[0], item[1], item[2])):
            if gi not in matched and pi not in used_predictions:
                matched[gi] = pi
                used_predictions.add(pi)

        decisions: list[OracleDecision] = []
        emitted_swap_pairs: set[tuple[int, int]] = set()
        for gi, gt in enumerate(gt_items):
            gt_id = int(gt["dataset_gt_id"])
            known_public = self._gt_to_public.get(gt_id)
            matched_pi = matched.get(gi)
            matched_public = None if matched_pi is None else pred_items[matched_pi].get("public_id")
            same_public_pi = None
            if known_public is not None:
                same_public = [
                    pi for pi, pred in enumerate(pred_items) if pred.get("public_id") == known_public
                ]
                same_public_pi = same_public[0] if same_public else None

            action: Optional[str] = None
            target_public = known_public
            other_public = None
            overlap_iou = None if matched_pi is None else float(overlaps[gi, matched_pi])
            confirmation = False
            if known_public is None:
                if matched_pi is None:
                    action = "ADD_NEW_IDENTITY"
                    target_public = None
                else:
                    # The runtime public ID is supplied explicitly to the
                    # oracle as a current-frame confirmation candidate; the
                    # oracle does not infer it from native/candidate IDs.
                    action = "ADD_NEW_IDENTITY"
                    target_public = None if matched_public is None else int(matched_public)
                    confirmation = matched_public is not None
            elif matched_pi is None:
                if same_public_pi is not None:
                    action = "AUTHORITATIVE_CORRECT"
                    overlap_iou = float(overlaps[gi, same_public_pi])
                else:
                    action = "RECOVER_IDENTITY"
            elif matched_public != known_public:
                action = "AUTHORITATIVE_REASSIGN"
                target_public = known_public
            elif overlap_iou is not None and overlap_iou < float(localization_iou_threshold):
                action = "AUTHORITATIVE_CORRECT"
            else:
                continue

            # Reciprocal known-ID mistakes are represented by one atomic swap
            # event, not two independently ordered reassignments.
            if action == "AUTHORITATIVE_REASSIGN" and matched_pi is not None:
                for other_gi, other_gt in enumerate(gt_items):
                    if other_gi == gi or other_gi not in matched:
                        continue
                    other_known = self._gt_to_public.get(int(other_gt["dataset_gt_id"]))
                    other_pi = matched[other_gi]
                    if other_known is None or pred_items[other_pi].get("public_id") != known_public:
                        continue
                    if matched_public == other_known:
                        pair = tuple(sorted((known_public, other_known)))
                        if pair not in emitted_swap_pairs:
                            emitted_swap_pairs.add(pair)
                            decisions.append(
                                OracleDecision(
                                    frame,
                                    "ATOMIC_ID_SWAP",
                                    gt_id,
                                    known_public,
                                    [float(x) for x in gt["box"]],
                                    int(matched_public),
                                    overlap_iou,
                                    other_public_id=int(other_known),
                                )
                            )
                        action = None
                        break
            if action is not None:
                decisions.append(
                    OracleDecision(
                        frame,
                        action,
                        gt_id,
                        None if target_public is None else int(target_public),
                        [float(x) for x in gt["box"]],
                        None if matched_public is None else int(matched_public),
                        overlap_iou,
                        other_public_id=other_public,
                        mapping_confirmation_required=confirmation,
                    )
                )

        known_output_public_ids = {int(value) for value in self._gt_to_public.values()}
        for pi, pred in enumerate(pred_items):
            if pi in used_predictions or pred.get("public_id") is None:
                continue
            public_id = int(pred["public_id"])
            if public_id in known_output_public_ids:
                decisions.append(
                    OracleDecision(
                        frame,
                        "AUTHORITATIVE_DELETE",
                        None,
                        public_id,
                        None,
                        public_id,
                        None,
                    )
                )
        self._events.extend(decisions)
        self.audit.event_count += len(decisions)
        self.audit.runtime_future_gt_used = bool(self.audit.future_frame_reads > 0)
        return decisions

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R3_SIMULATED_HUMAN_ORACLE_V1",
            "sequence": self.sequence,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "dataset_gt_to_public": dict(sorted(self._gt_to_public.items())),
            "audit": self.audit.as_dict(),
            "events": [event.as_dict() for event in self._events],
            "runtime_future_gt_used": bool(self.audit.future_frame_reads > 0),
        }


__all__ = ["ACTION_TYPES", "OracleDecision", "OracleAudit", "SimulatedHumanOracle"]
