"""GT-driven interaction simulator.

The simulator may use ground truth ONLY to generate the human input at the
current frame.  It never reads future GT, future detections or future
identity assignments.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.interaction.actions import (
    ActionType,
    HumanInteraction,
    InteractionResult,
    SystemContext,
)
from sam3_intermot.interaction.add import perform_add
from sam3_intermot.interaction.correct import perform_correct
from sam3_intermot.interaction.delete import perform_delete
from sam3_intermot.interaction.reassign import perform_reassign
from sam3_intermot.tracking.association import box_iou
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.leakage import LeakageReport


class EventType(str, Enum):
    NEW_TARGET_MISSED = "NEW_TARGET_MISSED"
    MASK_DRIFT = "MASK_DRIFT"
    MASK_MERGE = "MASK_MERGE"
    MASK_FRAGMENT = "MASK_FRAGMENT"
    WRONG_PERSON_PROPAGATION = "WRONG_PERSON_PROPAGATION"
    LOW_PRESENCE = "LOW_PRESENCE"
    DUPLICATE_TRACK = "DUPLICATE_TRACK"
    FALSE_TRACK = "FALSE_TRACK"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    DETECTION_UNAVAILABLE = "DETECTION_UNAVAILABLE"
    REAPPEARANCE = "REAPPEARANCE"


@dataclass
class SimulatedEvent:
    frame_idx: int
    event_type: EventType
    track_id: Optional[int] = None
    gt_id: Optional[int] = None
    detail: str = ""


@dataclass
class GTFrame:
    boxes: List[np.ndarray] = field(default_factory=list)
    gt_ids: List[int] = field(default_factory=list)


class GTView:
    """Frame-limited GT accessor; blocks any future-frame access."""

    def __init__(self, frames: Dict[int, GTFrame], current: int) -> None:
        self._frames = frames
        self._current = current

    def frame(self, frame_idx: int) -> GTFrame:
        if frame_idx != self._current:
            raise RuntimeError(
                f"future/past GT access blocked: requested {frame_idx}, current {self._current}"
            )
        return self._frames.get(frame_idx, GTFrame())


@dataclass
class SimulatorConfig:
    enabled_actions: set = field(
        default_factory=lambda: {ActionType.ADD, ActionType.CORRECT, ActionType.REASSIGN, ActionType.DELETE}
    )
    detection_interval: int = 5
    budget_per_100_frames: int = 0
    match_iou_threshold: float = 0.25
    low_presence_threshold: float = 0.3
    mask_drift_iou_threshold: float = 0.5
    enable_lineage_aware_add: bool = True
    enable_soft_delete: bool = True
    enable_atomic_reassign: bool = True
    enable_abstention: bool = True
    enable_guard: bool = True
    utility_threshold: float = 0.0
    abstain_stable_utility_penalty: float = 0.5


class SimulatedInteractionDriver:
    """Runs automatic tracking + interaction simulation over one sequence."""

    def __init__(
        self,
        backend,
        manager: TrackManager,
        lineages: IdentityLineageRegistry,
        config: Optional[SimulatorConfig] = None,
        detector: Optional[Callable[[int, str], List[PromptObjectObservation]]] = None,
    ) -> None:
        self.backend = backend
        self.manager = manager
        self.lineages = lineages
        self.config = config or SimulatorConfig()
        self.detector = detector or backend.detect_concept
        self.ctx = SystemContext(backend=backend, manager=manager, lineages=lineages)
        self.ctx.config.enable_lineage_aware_add = self.config.enable_lineage_aware_add
        self.ctx.config.enable_soft_delete = self.config.enable_soft_delete
        self.ctx.config.enable_atomic_reassign = self.config.enable_atomic_reassign
        self.ctx.config.enable_abstention = self.config.enable_abstention
        self.ctx.config.enable_guard = self.config.enable_guard
        self.ctx.config.utility_threshold = self.config.utility_threshold
        self.ctx.config.abstain_stable_utility_penalty = self.config.abstain_stable_utility_penalty
        self.events: List[SimulatedEvent] = []
        self.results: List[InteractionResult] = []
        self.leakage = LeakageReport()
        self._actions_used = 0
        self.abstentions = 0
        self.guard_rollbacks = []

    def run(
        self,
        gt_frames: Dict[int, GTFrame],
        num_frames: int,
        start_frame: int = 0,
    ) -> dict:
        max_actions = max(
            0,
            int(self.config.budget_per_100_frames * num_frames / 100.0),
        )
        for frame_idx in range(start_frame, num_frames):
            self._automatic_step(frame_idx)
            view = GTView(gt_frames, frame_idx)
            self._interaction_step(view.frame(frame_idx), frame_idx, max_actions)
        return {
            "events": self.events,
            "results": self.results,
            "actions_used": self._actions_used,
            "abstentions": self.abstentions,
            "guard_rollbacks": self.guard_rollbacks,
            "leakage": self.leakage,
        }

    # ------------------------------------------------------------------
    def _automatic_step(self, frame_idx: int) -> None:
        if frame_idx % self.config.detection_interval == 0:
            detections = self.detector(frame_idx, "person")
        else:
            detections = []
        propagated = (
            self.backend.propagate(frame_idx - 1, frame_idx)
            if frame_idx > 0
            else {}
        )
        observations = list(detections) + propagated.get(frame_idx, [])
        seen_sam = set()
        for obs in observations:
            if obs.sam_object_id in seen_sam:
                continue
            seen_sam.add(obs.sam_object_id)
            track = self._track_for_sam(obs.sam_object_id)
            if track is not None:
                self.manager.update_track(track.mot_track_id, frame_idx, obs)
            else:
                lineage = self.lineages.create(frame_idx)
                track = self.manager.create_track(frame_idx, obs, lineage.lineage_id)
                lineage.bind_track(track.mot_track_id)
        # mark unmatched active tracks
        matched_sam = {obs.sam_object_id for obs in observations}
        for track in self.manager.active_tracks():
            if track.sam_object_id is not None and track.sam_object_id not in matched_sam:
                self.manager.mark_missed(track.mot_track_id, frame_idx)

    def _track_for_sam(self, sam_object_id: int):
        for track in self.manager.active_tracks():
            if track.sam_object_id == sam_object_id:
                return track
        return None

    # ------------------------------------------------------------------
    def _interaction_step(
        self, gt: GTFrame, frame_idx: int, max_actions: int
    ) -> None:
        outputs = self.manager.outputs_for_frame(frame_idx)
        matched_outputs = set()
        for gt_id, gt_box in zip(gt.gt_ids, gt.boxes):
            best = None
            best_iou = self.config.match_iou_threshold
            for obs in outputs:
                if id(obs) in matched_outputs:
                    continue
                iou = box_iou(gt_box, obs.box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best = obs
            if best is None:
                self.events.append(
                    SimulatedEvent(
                        frame_idx=frame_idx,
                        event_type=EventType.NEW_TARGET_MISSED,
                        gt_id=gt_id,
                    )
                )
                self._maybe_add(gt_id, gt_box, frame_idx, max_actions)
            else:
                matched_outputs.add(id(best))
                track = self._track_for_sam(best.sam_object_id)
                if track is not None:
                    self._check_wrong_person(track, gt_id, frame_idx, max_actions)
                    self._check_presence(track, best, frame_idx)
        for obs in outputs:
            if id(obs) in matched_outputs:
                continue
            track = self._track_for_sam(obs.sam_object_id)
            if track is not None:
                self.events.append(
                    SimulatedEvent(
                        frame_idx=frame_idx,
                        event_type=EventType.FALSE_TRACK,
                        track_id=track.mot_track_id,
                    )
                )
                self._maybe_delete(track, frame_idx, max_actions)

    def _maybe_add(self, gt_id: int, gt_box: np.ndarray, frame_idx: int, max_actions: int) -> None:
        if ActionType.ADD not in self.config.enabled_actions or self._actions_used >= max_actions:
            return
        if self.config.enable_abstention and self._action_utility("Add") <= self.config.utility_threshold:
            self.abstentions += 1
            return
        action = HumanInteraction(
            action_id=str(uuid.uuid4()),
            frame_idx=frame_idx,
            action_type="Add",
            box_xyxy=gt_box,
            source="sim_gt_current_frame",
        )
        result = perform_add(self.ctx, action)
        self.results.append(result)
        self._guard_check(result)
        if result.accepted:
            self._actions_used += 1

    def _check_wrong_person(self, track, gt_id: int, frame_idx: int, max_actions: int) -> None:
        # First version: tracks carry a "current gt id" tag only for simulation
        # bookkeeping; a mismatch is flagged as Wrong Person and fixed with
        # Correct (or Reassign when the destination identity already exists).
        expected = getattr(track, "_sim_gt_id", None)
        if expected is not None and expected != gt_id:
            self.events.append(
                SimulatedEvent(
                    frame_idx=frame_idx,
                    event_type=EventType.WRONG_PERSON_PROPAGATION,
                    track_id=track.mot_track_id,
                    gt_id=gt_id,
                )
            )
            if self.config.enable_abstention:
                utility = self._action_utility(
                    "Correct", track=track,
                    obs=self.manager.outputs_for_frame(frame_idx)[0] if self.manager.outputs_for_frame(frame_idx) else None,
                )
                if utility <= self.config.utility_threshold:
                    self.abstentions += 1
                    setattr(track, "_sim_gt_id", gt_id)
                    return
            if ActionType.REASSIGN in self.config.enabled_actions and self._actions_used < max_actions:
                dest = self._find_track_by_gt_id(gt_id)
                if dest is not None and dest.mot_track_id != track.mot_track_id:
                    action = HumanInteraction(
                        action_id=str(uuid.uuid4()),
                        frame_idx=frame_idx,
                        action_type="Reassign",
                        target_track_id=track.mot_track_id,
                        destination_track_id=dest.mot_track_id,
                        source="sim_gt_current_frame",
                    )
                    result = perform_reassign(self.ctx, action)
                    self.results.append(result)
                    self._guard_check(result)
                    if result.accepted:
                        self._actions_used += 1
                    return
            if ActionType.CORRECT in self.config.enabled_actions and self._actions_used < max_actions:
                action = HumanInteraction(
                    action_id=str(uuid.uuid4()),
                    frame_idx=frame_idx,
                    action_type="Correct",
                    target_track_id=track.mot_track_id,
                    box_xyxy=track.last_box,
                    source="sim_gt_current_frame",
                )
                result = perform_correct(self.ctx, action)
                self.results.append(result)
                self._guard_check(result)
                if result.accepted:
                    self._actions_used += 1
        setattr(track, "_sim_gt_id", gt_id)

    def _check_presence(self, track, obs: PromptObjectObservation, frame_idx: int) -> None:
        if (
            obs.presence_score is not None
            and obs.presence_score < self.config.low_presence_threshold
        ):
            self.events.append(
                SimulatedEvent(
                    frame_idx=frame_idx,
                    event_type=EventType.LOW_PRESENCE,
                    track_id=track.mot_track_id,
                )
            )

    def _maybe_delete(self, track, frame_idx: int, max_actions: int) -> None:
        if ActionType.DELETE not in self.config.enabled_actions or self._actions_used >= max_actions:
            return
        if self.config.enable_abstention and self._action_utility("Delete", track=track) <= self.config.utility_threshold:
            self.abstentions += 1
            return
        action = HumanInteraction(
            action_id=str(uuid.uuid4()),
            frame_idx=frame_idx,
            action_type="Delete",
            target_track_id=track.mot_track_id,
            source="sim_gt_current_frame",
        )
        result = perform_delete(self.ctx, action)
        self.results.append(result)
        self._guard_check(result)
        if result.accepted:
            self._actions_used += 1

    def _action_utility(self, action_type: str, track=None, obs=None) -> float:
        if action_type == "Correct":
            utility = 1.0
        elif action_type == "Reassign":
            utility = 0.8
        elif action_type == "Delete":
            utility = 0.7
        else:
            utility = 0.5
        if track is not None:
            if track.confidence_history and np.mean(track.confidence_history) > 0.8:
                utility -= self.config.abstain_stable_utility_penalty
            if track.last_human_verified_frame is not None:
                utility -= 0.2
        return utility

    def _guard_check(self, result: InteractionResult) -> None:
        if not self.config.enable_guard or not result.accepted:
            return
        if result.action_type == "Correct":
            before_ids = {t["mot_track_id"] for t in result.before_summary.get("active_tracks", [])}
            after_ids = {t["mot_track_id"] for t in result.after_summary.get("active_tracks", [])}
            if before_ids != after_ids:
                self.guard_rollbacks.append({"action_id": result.action_id, "reason": "correct_changed_mot_ids"})
        if result.action_type == "Reassign":
            before_n = len(result.before_summary.get("active_tracks", []))
            after_n = len(result.after_summary.get("active_tracks", []))
            if after_n > before_n:
                self.guard_rollbacks.append({"action_id": result.action_id, "reason": "reassign_increased_ids"})
        if result.action_type == "Delete":
            before_ids = {t["mot_track_id"] for t in result.before_summary.get("active_tracks", [])}
            after_ids = {t["mot_track_id"] for t in result.after_summary.get("active_tracks", [])}
            if len(after_ids) > len(before_ids):
                self.guard_rollbacks.append({"action_id": result.action_id, "reason": "delete_recreated_id"})

    def _find_track_by_gt_id(self, gt_id: int):
        for track in self.manager.active_tracks():
            if getattr(track, "_sim_gt_id", None) == gt_id:
                return track
        return None
