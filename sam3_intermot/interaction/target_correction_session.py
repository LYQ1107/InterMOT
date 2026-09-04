"""Independent one-target SAM session used by N72R6.

The main SAM stream is never re-prompted.  This adapter owns one fresh
``Sam3Backend`` session, accepts only the event-frame human box, and exposes
official target-session observations separately from the backend's synthetic
human ledger observation.  It has no GT or public-ID inference path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation


def extract_human_roi_feature(
    image_path: str | Path,
    human_box: Sequence[float],
    encoder: Any,
) -> np.ndarray:
    """Extract a frozen OSNet feature from the raw current-frame human ROI.

    ``encoder`` is deliberately injected so this function cannot silently use
    a target-session SAM candidate feature as the human anchor.
    """

    box = np.asarray(human_box, dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("human_box must contain four finite coordinates")
    values = encoder.encode(Path(image_path), [box.tolist()])
    features = np.asarray(values, dtype=np.float32).reshape(-1, 512)
    if features.shape != (1, 512) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"human ROI encoder returned invalid shape/values: {features.shape}")
    norm = float(np.linalg.norm(features[0]))
    if norm <= 1.0e-6:
        raise RuntimeError("human ROI feature has zero norm")
    return features[0] / norm


@dataclass
class TargetScopedCorrectionSession:
    """One independent backend session containing exactly one target object."""

    backend: Any
    event_id: str
    sequence: str
    public_id: int
    event_frame: int
    target_object_id: int = 1
    target_session_scope: Optional[str] = None
    session_id: Optional[str] = None
    # ``frame_offset`` is zero for a full-sequence session.  An event-local
    # frame-range session may use a non-zero offset, but all public artifacts
    # remain in sequence-global coordinates.
    frame_offset: int = 0
    human_box: Optional[list[float]] = None
    seeded: bool = False
    main_y_pre_frozen: bool = False
    _official_by_frame: Dict[int, List[PromptObjectObservation]] = field(default_factory=dict)
    _recovery_audit: Optional[dict[str, Any]] = None
    _recovery_attempts: List[dict[str, Any]] = field(default_factory=list)
    _target_raw_sam_id: Optional[int] = None
    _official_extra_object_audit: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id)
        self.sequence = str(self.sequence)
        self.public_id = int(self.public_id)
        self.event_frame = int(self.event_frame)
        self.frame_offset = int(self.frame_offset)
        if self.public_id <= 0:
            raise ValueError("public_id must be positive")
        if self.target_session_scope is None:
            self.target_session_scope = f"n72r6-target:{self.event_id}:public:{self.public_id}"
        if self.frame_offset < 0 or self.event_frame < self.frame_offset:
            raise ValueError("frame_offset must be non-negative and no greater than event_frame")

    @property
    def session_event_frame(self) -> int:
        return int(self.event_frame - self.frame_offset)

    def _local_frame(self, global_frame: int) -> int:
        value = int(global_frame) - int(self.frame_offset)
        if value < 0:
            raise ValueError(f"global frame precedes target session range: {global_frame}")
        return value

    def _global_frame(self, local_frame: int) -> int:
        return int(local_frame) + int(self.frame_offset)

    def start(self, video_source: str | Path, *, main_y_pre_frozen: bool) -> str:
        if self._closed:
            raise RuntimeError("target correction session is closed")
        if self.session_id is not None:
            raise RuntimeError("target correction session already started")
        if not main_y_pre_frozen:
            raise ValueError("target session requires a frozen main Y_pre marker")
        self.main_y_pre_frozen = True
        self.session_id = str(self.backend.start_video(str(video_source)))
        return self.session_id

    @staticmethod
    def _copy_rows(rows: Sequence[PromptObjectObservation]) -> list[PromptObjectObservation]:
        return [row.copy() for row in rows]

    @staticmethod
    def _raw_id(observation: PromptObjectObservation) -> int:
        value = observation.raw_sam_object_id
        return int(observation.sam_object_id if value is None else value)

    def _target_only_rows(
        self,
        frame: int,
        rows: Sequence[PromptObjectObservation],
    ) -> list[PromptObjectObservation]:
        """Expose only the fixed official target raw ID.

        SAM3 partial propagation can merge fresh scene detections into an
        otherwise isolated tracker state.  Those rows are never allowed into
        the target candidate stream.  We retain a compact audit of the
        contamination and represent a missing target as an empty frame; we
        never relabel a competitor as the requested target.
        """

        copied = self._copy_rows(rows)
        if self._target_raw_sam_id is None:
            raise RuntimeError("target raw SAM ID is not established at the event frame")
        target_rows = [
            row for row in copied if self._raw_id(row) == int(self._target_raw_sam_id)
        ]
        extra_rows = [
            row for row in copied if self._raw_id(row) != int(self._target_raw_sam_id)
        ]
        if len(target_rows) > 1:
            raise RuntimeError(
                f"target session exposed duplicate target raw object: "
                f"{self.event_id}:{self._global_frame(int(frame))}:"
                f"{self._target_raw_sam_id}:{len(target_rows)}"
            )
        if extra_rows:
            self._official_extra_object_audit.append(
                {
                    "local_frame": int(frame),
                    "global_frame": self._global_frame(int(frame)),
                    "target_raw_sam_id": int(self._target_raw_sam_id),
                    "official_row_count": len(copied),
                    "target_row_count": len(target_rows),
                    "extra_raw_sam_ids": [self._raw_id(row) for row in extra_rows],
                    "extra_adapter_ids": [int(row.sam_object_id) for row in extra_rows],
                    "action": "EXCLUDE_NON_TARGET_OFFICIAL_ROWS",
                    "runtime_future_gt_used": False,
                }
            )
        return target_rows

    def seed_from_human_box(self, human_box: Sequence[float]) -> list[PromptObjectObservation]:
        if self.session_id is None:
            raise RuntimeError("start() must precede seed_from_human_box()")
        if not self.main_y_pre_frozen:
            raise RuntimeError("main Y_pre must be frozen before target correction")
        box = np.asarray(human_box, dtype=float).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError("human_box must contain four finite coordinates")
        self.human_box = box.astype(float).tolist()
        # The returned adapter observation is intentionally discarded: it is a
        # synthetic human-ledger row, not the target-session official output.
        prompt_frame = self.session_event_frame
        self.backend.add_box(prompt_frame, self.target_object_id, box)
        getter = getattr(self.backend, "get_last_official_prompt_outputs", None)
        if not callable(getter):
            raise RuntimeError("backend lacks explicit official prompt-output accessor")
        official = self._copy_rows(getter(prompt_frame))
        # A box-only prompt may return either no official row or several
        # detector rows.  Both cases require the same explicitly audited
        # official recovery; never expose a non-target scene row as target.
        if len(official) != 1:
            recover = getattr(self.backend, "recover_target_box_prompt", None)
            if not callable(recover):
                raise RuntimeError(
                    "official target-session box prompt was not singleton and no supported recovery exists"
                )
            recovery = recover(
                prompt_frame,
                self.target_object_id,
                box,
            )
            self._recovery_audit = dict(recovery)
            official = self._copy_rows(getter(prompt_frame))
        if len(official) != 1:
            raise RuntimeError(
                "target event prompt did not expose exactly one official target row: "
                f"{self.event_id}:{self.event_frame}:{len(official)}"
            )
        self._target_raw_sam_id = self._raw_id(official[0])
        self._official_by_frame[prompt_frame] = official
        self.seeded = True
        return self._copy_rows(official)

    def propagate_to(self, end_frame: int) -> dict[int, list[PromptObjectObservation]]:
        if not self.seeded or self.session_id is None:
            raise RuntimeError("seed_from_human_box() must precede propagation")
        end = int(end_frame)
        if end < self.event_frame:
            raise ValueError("end_frame must include the event frame")
        return self.propagate_from(self.event_frame, end)

    def propagate_from(self, start_frame: int, end_frame: int) -> dict[int, list[PromptObjectObservation]]:
        """Propagate a target-only suffix from an explicit global frame.

        This is used only by the target-session recovery probe.  A recovery
        prompt is seeded from the last already observed target-session box;
        it never receives a dataset GT box or any main-session observation.
        """

        if not self.seeded or self.session_id is None:
            raise RuntimeError("seed_from_human_box() must precede propagation")
        start = int(start_frame)
        end = int(end_frame)
        if start < self.event_frame or end < start:
            raise ValueError("propagation range must be within event_frame..end_frame")
        local_start = self._local_frame(start)
        local_end = self._local_frame(end)
        outputs = self.backend.propagate(
            local_start,
            local_end,
            start_frame_index=local_start,
            keep_masks=True,
            cache_outputs=True,
        )
        for frame, rows in outputs.items():
            frame_id = int(frame)
            copied = self._target_only_rows(frame_id, rows)
            # The official propagation iterator may emit an empty event-frame
            # cache row because the prompt response is stored separately.
            # Never let that transport row erase the official add-box result
            # captured by seed_from_human_box().
            if frame_id == self.session_event_frame and self._official_by_frame.get(frame_id):
                continue
            self._official_by_frame[frame_id] = copied
        return {
            self._global_frame(int(frame)): self._copy_rows(rows)
            for frame, rows in sorted(self._official_by_frame.items())
            if local_start <= int(frame) <= local_end
        }

    def recover_from_last_observation(
        self,
        frame: int,
        box_xyxy: Sequence[float],
        *,
        source_frame: int,
    ) -> Optional[PromptObjectObservation]:
        """Re-prompt only the target session from a past target observation.

        The box is intentionally supplied by the caller as a previously
        emitted target-session observation.  This method is not a human
        correction and cannot infer a public ID; its audit records the source
        frame so a posthoc validator can reject accidental future/GT input.
        """

        if not self.seeded or self.session_id is None:
            raise RuntimeError("seed_from_human_box() must precede target recovery")
        global_frame = int(frame)
        source = int(source_frame)
        if global_frame <= self.event_frame or source >= global_frame:
            raise ValueError("target recovery must use a strictly earlier target-session observation")
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError("recovery box must contain four finite coordinates")
        local_frame = self._local_frame(global_frame)
        response = self.backend.recover_target_box_prompt(
            local_frame,
            self.target_object_id,
            box,
            text="person",
        )
        getter = getattr(self.backend, "get_last_official_prompt_outputs", None)
        if not callable(getter):
            raise RuntimeError("backend lacks official recovery output accessor")
        official = self._copy_rows(getter(local_frame))
        previous_target_raw_sam_id = self._target_raw_sam_id
        retained_raw_sam_id = response.get("retained_raw_sam_id") if isinstance(response, dict) else None
        if retained_raw_sam_id is None:
            target_rows = self._target_only_rows(local_frame, official)
        else:
            # An official prompt may allocate a fresh raw SAM object after a
            # lost interval.  The backend has already isolated that returned
            # raw object; bind the target session to the audited new raw axis
            # instead of filtering it with the event-frame raw ID.
            retained_raw_sam_id = int(retained_raw_sam_id)
            target_rows = [
                row for row in official if self._raw_id(row) == retained_raw_sam_id
            ]
            if len(official) != 1 or len(target_rows) != 1:
                raise RuntimeError(
                    "target recovery official output is not a singleton retained raw object: "
                    f"{self.event_id}:{global_frame}:official={len(official)}:"
                    f"retained_raw={retained_raw_sam_id}:matched={len(target_rows)}"
                )
        if len(target_rows) != 1:
            raise RuntimeError(
                f"target recovery did not expose exactly one target row: "
                f"{self.event_id}:{global_frame}:{len(target_rows)}"
            )
        self._target_raw_sam_id = self._raw_id(target_rows[0])
        self._official_by_frame[local_frame] = target_rows
        audit = {
            "global_frame": global_frame,
            "local_frame": local_frame,
            "source_frame": source,
            "source": "last_observed_target_session_box",
            "seed_box_xyxy": box.astype(float).tolist(),
            "official_response": deepcopy(response),
            "previous_target_raw_sam_id": (
                None
                if previous_target_raw_sam_id is None
                else int(previous_target_raw_sam_id)
            ),
            "recovered_raw_sam_id": int(self._target_raw_sam_id),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
        self._recovery_attempts.append(audit)
        return target_rows[0].copy()

    def record_recovery_failure(
        self,
        frame: int,
        box_xyxy: Sequence[float],
        *,
        source_frame: int,
        error: BaseException,
    ) -> None:
        """Record a legal target loss after official recovery has no output.

        N72R6 defines a missing target candidate as target-public ``NONE`` /
        ``LOST``; inability to recover must therefore remain an audited
        outcome rather than aborting the complete frame stream.  The backend
        supplies the prompt-attempt audit, while this session adds the causal
        source-frame provenance.  No candidate or identity is synthesized.
        """

        global_frame = int(frame)
        source = int(source_frame)
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        if global_frame <= self.event_frame or source >= global_frame:
            raise ValueError("recovery failure must use a strictly earlier target-session observation")
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError("recovery failure box must contain four finite coordinates")
        response = getattr(self.backend, "_last_recovery_failure", None)
        if not isinstance(response, dict):
            response = {
                "schema_version": "N72R6_TARGET_RECOVERY_FAILURE_V1",
                "status": "FAIL_TARGET_RECOVERY_NO_OFFICIAL_OBSERVATION",
                "frame_idx": self._local_frame(global_frame),
                "retained_official_count": 0,
                "retained_raw_sam_id": None,
                "runtime_future_gt_used": False,
            }
        audit = {
            "global_frame": global_frame,
            "local_frame": self._local_frame(global_frame),
            "source_frame": source,
            "source": "last_observed_target_session_box",
            "seed_box_xyxy": box.astype(float).tolist(),
            "status": "FAIL_TARGET_RECOVERY_NO_OFFICIAL_OBSERVATION",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_response": deepcopy(response),
            "recovered_raw_sam_id": None,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
        self._recovery_attempts.append(audit)

    def candidate_at(self, frame: int) -> Optional[PromptObjectObservation]:
        rows = self._official_by_frame.get(self._local_frame(int(frame)), [])
        if len(rows) > 1:
            raise RuntimeError(
                f"target session exposed more than one object: {self.event_id}:{frame}:{len(rows)}"
            )
        return None if not rows else rows[0].copy()

    def audit(self) -> dict[str, Any]:
        capacity = {
            "max_num_objects": getattr(self.backend, "max_num_objects", None),
            "multiplex_count": getattr(self.backend, "multiplex_count", None),
            "requested_object_count": 1,
            "observed_max_objects_per_frame": max(
                [len(rows) for rows in self._official_by_frame.values()] or [0]
            ),
        }
        return {
            "schema_version": "N72R6_TARGET_SESSION_AUDIT_V1",
            "event_id": self.event_id,
            "sequence": self.sequence,
            "public_id": self.public_id,
            "event_frame": self.event_frame,
            "frame_offset": self.frame_offset,
            "session_event_frame": self.session_event_frame,
            "session_id": self.session_id,
            "target_object_id": self.target_object_id,
            "target_session_scope": self.target_session_scope,
            "main_y_pre_frozen": self.main_y_pre_frozen,
            "seeded_from_human_box": self.seeded,
            "official_frame_count": len(self._official_by_frame),
            "official_candidate_count": sum(len(rows) for rows in self._official_by_frame.values()),
            "target_raw_sam_id": self._target_raw_sam_id,
            "official_extra_object_count": len(self._official_extra_object_audit),
            "official_extra_object_audit": deepcopy(self._official_extra_object_audit),
            "capacity": capacity,
            "event_frame_memory_read": False,
            "first_memory_visible_frame": self.event_frame + 1,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "public_id_inference": False,
            "recovery_audit": deepcopy(self._recovery_audit),
            "recovery_attempts": deepcopy(self._recovery_attempts),
            "closed": self._closed,
            "prompt_fallback_log": deepcopy(getattr(self.backend, "_prompt_fallback_log", [])),
            "resume_repair_log": deepcopy(getattr(self.backend, "_resume_repair_log", [])),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.backend.close()
        finally:
            self.session_id = None
            self._official_by_frame.clear()
            self._closed = True


__all__ = ["TargetScopedCorrectionSession", "extract_human_roi_feature"]
