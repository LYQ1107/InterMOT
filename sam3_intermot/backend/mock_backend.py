"""Deterministic mock backend for unit tests.

The mock backend is a test double only.  It produces synthetic observations
from prompts and is never used as a substitute for real SAM 3.1 results.
"""

import uuid
from typing import Dict, List, Optional

import numpy as np

from sam3_intermot.backend.base import PromptVideoTrackerBackend
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.observations.mask_to_box import mask_to_box


class MockBackend(PromptVideoTrackerBackend):
    """Simple deterministic tracker used exclusively in unit tests."""

    def __init__(
        self,
        frame_h: int = 720,
        frame_w: int = 1280,
        seed: int = 0,
        velocity_scale: float = 2.0,
    ) -> None:
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.seed = seed
        self.velocity_scale = velocity_scale
        self.video_source: Optional[str] = None
        self._objects: Dict[int, dict] = {}
        self._frame_outputs: Dict[int, List[PromptObjectObservation]] = {}
        self._output_cache: Dict[int, List[PromptObjectObservation]] = {}
        self._next_object_id = 1
        self._session_id: Optional[str] = None
        self._closed = False
        self._concept_boxes: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    def start_video(self, video_source: str) -> str:
        self.video_source = video_source
        self._objects = {}
        self._frame_outputs = {}
        self._next_object_id = 1
        self._closed = False
        self._session_id = f"mock-{uuid.uuid4().hex[:12]}"
        return self._session_id

    def set_concept_boxes(self, text_prompt: str, boxes: List[np.ndarray]) -> None:
        """Register deterministic concept-detection candidates for tests."""
        self._concept_boxes[text_prompt] = [np.asarray(b, dtype=float) for b in boxes]

    # ------------------------------------------------------------------
    def detect_concept(
        self, frame_idx: int, text_prompt: str
    ) -> List[PromptObjectObservation]:
        results: List[PromptObjectObservation] = []
        candidates = self._concept_boxes.get(text_prompt, [])
        for box in candidates:
            oid = self._match_existing(box)
            if oid is None:
                oid = self._allocate_id()
                self._objects[oid] = self._make_object(
                    frame_idx=frame_idx,
                    box=box,
                    source="concept_detection",
                    verified=False,
                )
            results.append(self._emit(frame_idx, oid, "concept_detection"))
        return results

    def add_box(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: np.ndarray,
    ) -> PromptObjectObservation:
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        if object_id not in self._objects:
            self._objects[object_id] = self._make_object(
                frame_idx=frame_idx,
                box=box,
                source="human_add",
                verified=True,
            )
        else:
            self._update_prompt(object_id, frame_idx, box)
        return self._emit(frame_idx, object_id, "human_add", verified=True)

    def add_points(
        self,
        frame_idx: int,
        object_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> PromptObjectObservation:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        pad = 5.0
        x1, y1 = pts.min(axis=0) - pad
        x2, y2 = pts.max(axis=0) + pad
        box = np.asarray([x1, y1, x2, y2], dtype=float)
        return self.add_box(frame_idx, object_id, box)

    def add_mask(
        self,
        frame_idx: int,
        object_id: int,
        mask: np.ndarray,
    ) -> PromptObjectObservation:
        box = mask_to_box(mask)
        if box is None:
            raise ValueError("empty mask has no box")
        return self.add_box(frame_idx, object_id, box)

    def correct_object(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> PromptObjectObservation:
        if object_id not in self._objects:
            raise ValueError(f"invalid object id: {object_id}")
        if mask is not None:
            box = mask_to_box(mask)
            if box is None:
                raise ValueError("empty mask cannot correct object")
        elif box_xyxy is not None:
            box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        elif points is not None:
            return self.add_points(frame_idx, object_id, points, labels or np.ones(len(points)))
        else:
            raise ValueError("correct_object requires box, points or mask")
        self._update_prompt(object_id, frame_idx, box)
        return self._emit(frame_idx, object_id, "human_correction", verified=True)

    def propagate(
        self,
        start_frame: int,
        end_frame: int,
        start_frame_index: Optional[int] = None,
        *,
        keep_masks: bool = True,
    ) -> dict:
        outputs: dict = {}
        for frame_idx in range(start_frame, end_frame + 1):
            frame_obs: List[PromptObjectObservation] = []
            for oid in sorted(self._objects):
                frame_obs.append(self._emit(frame_idx, oid, "automatic_propagation"))
            existing_human = {
                o.sam_object_id: o
                for o in self._frame_outputs.get(frame_idx, [])
                if o.is_human_verified
            }
            merged: Dict[int, PromptObjectObservation] = {}
            for o in frame_obs:
                merged[o.sam_object_id] = o
            for oid, o in existing_human.items():
                merged[oid] = o
            ordered = [merged[k] for k in sorted(merged)]
            self._frame_outputs[frame_idx] = ordered
            outputs[frame_idx] = [o.copy() for o in ordered]
        return outputs

    def remove_object(self, object_id: int) -> None:
        if object_id not in self._objects:
            raise ValueError(f"invalid object id: {object_id}")
        del self._objects[object_id]
        for frame_obs in self._frame_outputs.values():
            frame_obs[:] = [o for o in frame_obs if o.sam_object_id != object_id]

    def reset_object(self, object_id: int) -> None:
        if object_id not in self._objects:
            raise ValueError(f"invalid object id: {object_id}")
        obj = self._objects[object_id]
        obj["prompt_box"] = obj["initial_box"].copy()
        obj["prompt_frame"] = obj["initial_frame"]

    def reset_session(self) -> None:
        """Test double for the official whole-session reset primitive."""
        self._objects = {}
        self._frame_outputs = {}
        self._output_cache = {}
        self._next_object_id = 1

    def get_frame_outputs(self, frame_idx: int) -> List[PromptObjectObservation]:
        return [o.copy() for o in self._frame_outputs.get(frame_idx, [])]

    def close(self) -> None:
        self._objects.clear()
        self._frame_outputs.clear()
        self._closed = True
        self._session_id = None

    # ------------------------------------------------------------------
    def _allocate_id(self) -> int:
        oid = self._next_object_id
        self._next_object_id += 1
        return oid

    def _match_existing(self, box: np.ndarray) -> Optional[int]:
        for oid, obj in self._objects.items():
            cur = obj["prompt_box"]
            if _iou(box, cur) >= 0.5:
                return oid
        return None

    def _make_object(self, frame_idx: int, box: np.ndarray, source: str, verified: bool) -> dict:
        rng = np.random.default_rng(self.seed + frame_idx * 31 + int(box.sum()) % 1000)
        dx = float(rng.uniform(-self.velocity_scale, self.velocity_scale))
        dy = float(rng.uniform(-self.velocity_scale, self.velocity_scale))
        return {
            "initial_box": box.copy(),
            "initial_frame": frame_idx,
            "prompt_box": box.copy(),
            "prompt_frame": frame_idx,
            "velocity": np.asarray([dx, dy, dx, dy]),
            "source": source,
            "verified": verified,
            "confidence": 0.99,
            "presence": 0.95,
        }

    def _update_prompt(self, object_id: int, frame_idx: int, box: np.ndarray) -> None:
        obj = self._objects[object_id]
        obj["prompt_box"] = np.clip(box, 0, [self.frame_w, self.frame_h, self.frame_w, self.frame_h])
        obj["prompt_frame"] = frame_idx

    def _emit(
        self,
        frame_idx: int,
        object_id: int,
        source: str,
        verified: bool = False,
    ) -> PromptObjectObservation:
        obj = self._objects[object_id]
        box = obj["prompt_box"].copy()
        if obj["prompt_frame"] is not None and frame_idx > obj["prompt_frame"]:
            dt = frame_idx - obj["prompt_frame"]
            box = box + dt * obj["velocity"]
        box = np.clip(box, 0, [self.frame_w, self.frame_h, self.frame_w, self.frame_h])
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            box = np.asarray([x1, y1, x1 + 1, y2 + 1], dtype=float)
            x1, y1, x2, y2 = box
        mask = np.zeros((self.frame_h, self.frame_w), dtype=bool)
        mask[int(y1) : int(y2), int(x1) : int(x2)] = True
        return PromptObjectObservation(
            frame_idx=frame_idx,
            sam_object_id=object_id,
            mask=mask,
            box_xyxy=box,
            confidence=float(obj["confidence"]),
            presence_score=float(obj["presence"]),
            source=source,
            is_human_verified=verified,
        )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
