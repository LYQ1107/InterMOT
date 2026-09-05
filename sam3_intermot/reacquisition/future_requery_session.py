"""Causal, event-local SAM3 future-frame re-query session for N72R10.

This module deliberately does not infer or assign public identities.  It starts
an independent SAM3 target session at a future trigger frame, queries a small
frozen family of boxes derived from the *causal predicted* target box, and
exposes only raw/native candidate evidence.  Probe sessions are closed before
the next query; the selected query is then rerun in one fresh session for
future propagation.

The implementation is an adapter around the existing official backend and
``TargetScopedCorrectionSession``.  It never reads GT, future labels, or
posthoc metrics, and it does not clear official predictor state while a
selected propagation is active.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sam3_intermot.interaction.target_correction_session import TargetScopedCorrectionSession
from sam3_intermot.observations.mask_to_box import mask_to_box
from sam3_intermot.reacquisition.target_candidate_pool import (
    FEATURE_DIM,
    FUTURE_FRAME_REQUERY,
)


FUTURE_FRAME_REQUERY_CANDIDATE_KIND = "FUTURE_FRAME_REQUERY_CANDIDATE"
QUERY_SPECS: tuple[dict[str, float | str], ...] = (
    {
        "name": "PREDICTED_CENTER",
        "dx_fraction": 0.0,
        "dy_fraction": 0.0,
        "scale": 1.0,
    },
    {
        "name": "PREDICTED_SHRINK",
        "dx_fraction": 0.0,
        "dy_fraction": 0.0,
        "scale": 0.82,
    },
    {
        "name": "PREDICTED_LEFT",
        "dx_fraction": -0.24,
        "dy_fraction": 0.0,
        "scale": 1.0,
    },
    {
        "name": "PREDICTED_RIGHT",
        "dx_fraction": 0.24,
        "dy_fraction": 0.0,
        "scale": 1.0,
    },
)


def _json_safe(value: Any) -> Any:
    """Copy causal metadata without retaining tensor/array ownership."""

    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("causal state contains non-finite array values")
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("causal state contains a non-finite float")
        return float(value)
    raise TypeError(f"causal state contains unsupported value type: {type(value).__name__}")


def _reject_future_or_gt_metadata(value: Any, path: str = "causal_state") -> None:
    """Reject labels/metrics that would make this a posthoc-conditioned run."""

    forbidden_tokens = ("ground_truth", "gt_", "future_gt", "posthoc", "future_label", "future_iou")
    explicit_audit_flags = {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in explicit_audit_flags:
                if bool(item):
                    raise ValueError(f"nonzero forbidden runtime flag: {path}.{key}")
                continue
            if any(token in key_text for token in forbidden_tokens):
                raise ValueError(f"forbidden runtime metadata key: {path}.{key}")
            _reject_future_or_gt_metadata(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_future_or_gt_metadata(item, f"{path}[{index}]")


def _finite_box(value: Sequence[float], label: str) -> list[float]:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError(f"{label} must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} must have positive area")
    return [float(item) for item in box]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f4").reshape(-1).tobytes()).hexdigest()


def query_box(predicted_box: Sequence[float], spec: Mapping[str, Any]) -> list[float]:
    """Create a frozen prompt from the causal predicted box, never the event GT box."""

    box = np.asarray(_finite_box(predicted_box, "predicted_box"), dtype=np.float64)
    x1, y1, x2, y2 = [float(item) for item in box]
    width = x2 - x1
    height = y2 - y1
    scale = float(spec["scale"])
    dx = float(spec["dx_fraction"]) * width
    dy = float(spec["dy_fraction"]) * height
    cx = (x1 + x2) / 2.0 + dx
    cy = (y1 + y2) / 2.0 + dy
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0
    result = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    return _finite_box(result, f"query_box:{spec.get('name', 'unknown')}")


class FutureFrameRequerySession:
    """One causal future-frame re-query with strict session lifecycle."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], Any],
        sequence: str,
        event_id: str,
        event_frame: int,
        target_public_id: int,
        frame_paths: Sequence[str | Path] | Mapping[int, str | Path],
        feature_fn: Callable[[int, Sequence[float]], Any] | None = None,
        session_factory: Callable[..., TargetScopedCorrectionSession] = TargetScopedCorrectionSession,
        query_specs: Sequence[Mapping[str, Any]] = QUERY_SPECS,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable and return a fresh backend")
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self.backend_factory = backend_factory
        self.sequence = str(sequence)
        self.event_id = str(event_id)
        self.event_frame = int(event_frame)
        self.target_public_id = int(target_public_id)
        if self.target_public_id <= 0:
            raise ValueError("target_public_id must be positive")
        self.frame_paths = frame_paths
        self.feature_fn = feature_fn
        self.session_factory = session_factory
        self.query_specs = tuple(deepcopy(dict(spec)) for spec in query_specs)
        if not self.query_specs:
            raise ValueError("at least one frozen query spec is required")
        self._trigger_frame: int | None = None
        self._end_frame: int | None = None
        self._predicted_box: list[float] | None = None
        self._causal_state: dict[str, Any] = {}
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._window_path: Path | None = None
        self._window_mapping: list[dict[str, Any]] = []
        self._probe_candidates: list[dict[str, Any]] = []
        self._query_audits: list[dict[str, Any]] = []
        self._future_candidates: list[dict[str, Any]] = []
        self._future_frame_coverage: list[dict[str, Any]] = []
        self._invalid_observation_audit: list[dict[str, Any]] = []
        self._selected_query_name: str | None = None
        self._selected_candidate_uid: str | None = None
        self._selection_audit: dict[str, Any] | None = None
        self._active_backend: Any | None = None
        self._active_session: TargetScopedCorrectionSession | None = None
        self._active_session_audit: dict[str, Any] | None = None
        self._closed = False

    @property
    def trigger_frame(self) -> int:
        if self._trigger_frame is None:
            raise RuntimeError("start_from_frame() has not been called")
        return int(self._trigger_frame)

    @property
    def end_frame(self) -> int:
        if self._end_frame is None:
            raise RuntimeError("start_from_frame() has not been called")
        return int(self._end_frame)

    @staticmethod
    def _path_for_frame(
        frame_paths: Sequence[str | Path] | Mapping[int, str | Path],
        frame: int,
    ) -> Path:
        try:
            raw = frame_paths[frame]  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            raise ValueError(f"frame path is unavailable for global frame {frame}") from None
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"frame path is not a regular file: {path}")
        return path

    def start_from_frame(
        self,
        trigger_frame: int,
        predicted_box: Sequence[float],
        causal_state: Mapping[str, Any],
        *,
        end_frame: int | None = None,
        main_y_pre_frozen: bool = True,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("future re-query session is closed")
        if self._trigger_frame is not None:
            raise RuntimeError("future re-query session has already started")
        trigger = int(trigger_frame)
        end = self.event_frame + 100 if end_frame is None else int(end_frame)
        if trigger <= self.event_frame:
            raise ValueError("future re-query trigger must be strictly after event_frame")
        if end < trigger or end > self.event_frame + 100:
            raise ValueError("future re-query window must be trigger..event_frame+100")
        if not main_y_pre_frozen:
            raise ValueError("future re-query requires frozen main Y_pre")
        if not isinstance(causal_state, Mapping):
            raise TypeError("causal_state must be a mapping")
        _reject_future_or_gt_metadata(causal_state)
        box = _finite_box(predicted_box, "predicted_box")
        safe_state = _json_safe(causal_state)
        if not isinstance(safe_state, dict):
            raise TypeError("causal_state could not be serialized as a mapping")
        self._trigger_frame = trigger
        self._end_frame = end
        self._predicted_box = box
        self._causal_state = safe_state
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="n72r10_future_requery_")
        self._window_path = Path(self._temporary_directory.name) / "frames"
        self._window_path.mkdir(parents=True, exist_ok=True)
        self._window_mapping = []
        for local_frame, global_frame in enumerate(range(trigger, end + 1)):
            source = self._path_for_frame(self.frame_paths, global_frame)
            destination = self._window_path / f"{local_frame:06d}{source.suffix.lower() or '.jpg'}"
            os.symlink(str(source), str(destination))
            self._window_mapping.append(
                {
                    "local_frame": int(local_frame),
                    "global_frame": int(global_frame),
                    "source_path": str(source),
                    "local_path": str(destination),
                    "frame_sha256": _sha256_file(source),
                }
            )
        return {
            "status": "PASS_STARTED",
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_frame": int(self.event_frame),
            "trigger_frame": int(trigger),
            "end_frame": int(end),
            "local_frame_zero_global": int(trigger),
            "window_path": str(self._window_path),
            "window_mapping": deepcopy(self._window_mapping),
            "predicted_box_xyxy": list(box),
            "causal_state_sha256": hashlib.sha256(
                json.dumps(safe_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "main_y_pre_frozen": True,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }

    def _new_session(self, query_name: str) -> tuple[Any, TargetScopedCorrectionSession]:
        if self._window_path is None:
            raise RuntimeError("start_from_frame() must precede query")
        backend = self.backend_factory()
        if backend is None:
            raise RuntimeError("backend_factory returned None")
        session: TargetScopedCorrectionSession | None = None
        try:
            session = self.session_factory(
                backend=backend,
                event_id=f"{self.event_id}:future:{query_name}",
                sequence=self.sequence,
                public_id=self.target_public_id,
                event_frame=self.trigger_frame,
                target_session_scope=f"n72r10-future:{self.event_id}:{query_name}",
                frame_offset=self.trigger_frame,
                isolate_official_target_state=True,
                preserve_official_action_history=True,
            )
            session.start(self._window_path, main_y_pre_frozen=True)
            return backend, session
        except Exception:
            self._cleanup_backend_session(session, backend)
            raise

    @staticmethod
    def _raw_id(observation: Any) -> int:
        value = getattr(observation, "raw_sam_object_id", None)
        if value is None:
            value = getattr(observation, "sam_object_id", None)
        if value is None:
            raise ValueError("official observation has no raw/native object ID")
        return int(value)

    def _observation_feature(self, frame: int, box: Sequence[float]) -> tuple[list[float] | None, str | None]:
        if self.feature_fn is None:
            return None, None
        value = np.asarray(self.feature_fn(int(frame), list(box)), dtype=np.float32).reshape(-1)
        if value.size != FEATURE_DIM or not np.all(np.isfinite(value)):
            raise ValueError("feature_fn must return a finite 512-D feature")
        if float(np.linalg.norm(value)) <= 1.0e-6:
            raise ValueError("feature_fn returned a zero-norm feature")
        return value.astype(float).tolist(), _feature_sha256(value)

    def _serialize_observation(
        self,
        observation: Any,
        *,
        frame: int,
        query_index: int,
        query_name: str,
        query_box_xyxy: Sequence[float],
        box_override: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        raw_id = self._raw_id(observation)
        adapter_id = getattr(observation, "sam_object_id", raw_id)
        box = _finite_box(
            getattr(observation, "box_xyxy") if box_override is None else box_override,
            "official observation box",
        )
        feature, feature_hash = self._observation_feature(frame, box)
        mask = getattr(observation, "mask", None)
        mask_hash = None
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.size:
                mask_hash = hashlib.sha256(mask_array.astype(bool).tobytes()).hexdigest()
        uid = (
            f"{self.event_id}:future_frame_requery:{self.trigger_frame}:"
            f"{query_index}:{query_name}:{int(frame)}:{raw_id}:{int(adapter_id)}"
        )
        confidence = float(getattr(observation, "confidence", 0.0))
        presence = getattr(observation, "presence_score", None)
        presence_value = None if presence is None else float(presence)
        if not math.isfinite(confidence) or (presence_value is not None and not math.isfinite(presence_value)):
            raise ValueError("official observation confidence is non-finite")
        return {
            "candidate_uid": uid,
            "candidate_index": int(query_index),
            "candidate_kind": FUTURE_FRAME_REQUERY_CANDIDATE_KIND,
            "candidate_source": FUTURE_FRAME_REQUERY,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_frame": int(self.event_frame),
            "trigger_frame": int(self.trigger_frame),
            "frame": int(frame),
            "frame_horizon_from_trigger": int(frame - self.trigger_frame),
            "official_raw_sam_id": raw_id,
            "adapter_external_id": int(adapter_id),
            "native_tid": int(adapter_id),
            "native_scope": None,
            "native_tid_scope": None,
            "box_xyxy": box,
            "mask_sha256": mask_hash,
            "confidence": confidence,
            "presence_score": presence_value,
            "feature": feature,
            "feature_dim": None if feature is None else FEATURE_DIM,
            "feature_sha256": feature_hash,
            "feature_source": "future_frame_requery_machine_roi_feature" if feature is not None else None,
            "source": str(getattr(observation, "source", "official_future_requery")),
            "source_session_id": None,
            "target_session_scope": None,
            "requery_index": int(query_index),
            "requery_name": query_name,
            "requery_box_xyxy": [float(value) for value in query_box_xyxy],
            "public_id": None,
            "public_id_authority": None,
            "public_id_inference": False,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }

    def _serialize_propagated_observation(
        self,
        observation: Any,
        *,
        frame: int,
        query_index: int,
        query_name: str,
        query_box_xyxy: Sequence[float],
    ) -> dict[str, Any] | None:
        """Serialize one official future row, preserving an explicit absence.

        The pinned SAM3 response can retain an object ID while returning a
        finite zero-area box and an empty mask when the target is absent on a
        frame.  That is a missing official observation, not a candidate and
        must not be converted into a synthetic box.  Non-finite geometry or a
        non-empty mask with zero-area geometry remains an actionable failure.
        """

        raw_box = np.asarray(getattr(observation, "box_xyxy"), dtype=np.float64).reshape(-1)
        if raw_box.size != 4 or not np.all(np.isfinite(raw_box)):
            raise ValueError("official observation box must contain four finite coordinates")
        mask = np.asarray(getattr(observation, "mask", None))
        mask_nonempty = bool(mask.size and np.any(mask.astype(bool)))
        if raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]:
            if mask_nonempty:
                repaired_box = mask_to_box(mask.astype(bool))
                if repaired_box is None:
                    raise ValueError("official observation has a non-empty mask but no recoverable box")
                row = self._serialize_observation(
                    observation,
                    frame=frame,
                    query_index=query_index,
                    query_name=query_name,
                    query_box_xyxy=query_box_xyxy,
                    box_override=repaired_box,
                )
                self._invalid_observation_audit.append(
                    {
                        "frame": int(frame),
                        "query_index": int(query_index),
                        "query_name": str(query_name),
                        "raw_sam_object_id": int(self._raw_id(observation)),
                        "adapter_external_id": int(getattr(observation, "sam_object_id", self._raw_id(observation))),
                        "official_box_xyxy_raw": raw_box.astype(float).tolist(),
                        "repaired_box_xyxy": [float(value) for value in repaired_box],
                        "mask_shape": [int(value) for value in mask.shape],
                        "mask_present": True,
                        "mask_nonempty": True,
                        "status": "REPAIRED_BOX_FROM_OFFICIAL_NONEMPTY_MASK",
                        "action": "USE_DETERMINISTIC_MASK_TO_BOX_NO_SYNTHETIC_MASK",
                        "runtime_future_gt_used": False,
                        "runtime_gt_read": False,
                        "posthoc_gt_used": False,
                    }
                )
                row["official_box_xyxy_raw"] = raw_box.astype(float).tolist()
                row["box_provenance"] = "DETERMINISTIC_OFFICIAL_MASK_TO_BOX_REPAIR"
                return row
            self._invalid_observation_audit.append(
                {
                    "frame": int(frame),
                    "query_index": int(query_index),
                    "query_name": str(query_name),
                    "raw_sam_object_id": int(self._raw_id(observation)),
                    "adapter_external_id": int(getattr(observation, "sam_object_id", self._raw_id(observation))),
                    "official_box_xyxy_raw": raw_box.astype(float).tolist(),
                    "mask_shape": [int(value) for value in mask.shape],
                    "mask_present": bool(mask.size),
                    "mask_nonempty": False,
                    "status": "LEGITIMATELY_ABSENT_OFFICIAL_ZERO_AREA_EMPTY_MASK",
                    "action": "EXCLUDE_FROM_CANDIDATE_STREAM_NO_SYNTHETIC_BOX",
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                }
            )
            return None
        return self._serialize_observation(
            observation,
            frame=frame,
            query_index=query_index,
            query_name=query_name,
            query_box_xyxy=query_box_xyxy,
        )

    @staticmethod
    def _cleanup_backend_session(
        session: TargetScopedCorrectionSession | None,
        backend: Any | None,
    ) -> None:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        elif backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        del session
        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def query_current_frame(self) -> list[dict[str, Any]]:
        """Run every frozen query in a fresh backend and return trigger rows."""

        if self._trigger_frame is None:
            raise RuntimeError("start_from_frame() must precede query_current_frame()")
        if self._probe_candidates or self._query_audits:
            raise RuntimeError("query_current_frame() is single-use")
        for query_index, spec in enumerate(self.query_specs):
            query_name = str(spec["name"])
            prompt_box = query_box(self._predicted_box or [], spec)
            backend: Any | None = None
            session: TargetScopedCorrectionSession | None = None
            try:
                backend, session = self._new_session(query_name)
                session.seed_from_human_box(prompt_box)
                observation = session.candidate_at(self.trigger_frame)
                if observation is None:
                    raise RuntimeError("official query returned no target candidate at trigger frame")
                row = self._serialize_observation(
                    observation,
                    frame=self.trigger_frame,
                    query_index=query_index,
                    query_name=query_name,
                    query_box_xyxy=prompt_box,
                )
                row["source_session_id"] = session.session_id
                row["target_session_scope"] = session.target_session_scope
                row["native_scope"] = session.target_session_scope
                row["native_tid_scope"] = session.target_session_scope
                self._probe_candidates.append(row)
                session_audit = session.audit()
                self._query_audits.append(
                    {
                        "status": "PASS_QUERY_CURRENT_FRAME",
                        "query_index": int(query_index),
                        "query_name": query_name,
                        "query_spec": deepcopy(dict(spec)),
                        "query_box_xyxy": prompt_box,
                        "candidate_uid": row["candidate_uid"],
                        "candidate_count": 1,
                        "target_session_audit": session_audit,
                        "probe_then_rerun_active_session": True,
                        "runtime_future_gt_used": False,
                        "runtime_gt_read": False,
                        "posthoc_gt_used": False,
                    }
                )
            except Exception as exc:
                self._query_audits.append(
                    {
                        "status": "FAIL_QUERY_CURRENT_FRAME",
                        "query_index": int(query_index),
                        "query_name": query_name,
                        "query_spec": deepcopy(dict(spec)),
                        "query_box_xyxy": prompt_box,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "candidate_count": 0,
                        "runtime_future_gt_used": False,
                        "runtime_gt_read": False,
                        "posthoc_gt_used": False,
                    }
                )
            finally:
                self._cleanup_backend_session(session, backend)
        return deepcopy(self._probe_candidates)

    def candidate_pool(self) -> list[dict[str, Any]]:
        if self._trigger_frame is None:
            raise RuntimeError("start_from_frame() must precede candidate_pool()")
        return deepcopy(self._probe_candidates)

    def _find_query(self, selected_query_name: str | None, selected_candidate_uid: str | None) -> tuple[int, dict[str, Any]]:
        matches = list(self._probe_candidates)
        if selected_candidate_uid is not None:
            matches = [row for row in matches if str(row["candidate_uid"]) == str(selected_candidate_uid)]
        if selected_query_name is not None:
            matches = [row for row in matches if str(row["requery_name"]) == str(selected_query_name)]
        if len(matches) != 1:
            raise ValueError("selection must identify exactly one successful future re-query candidate")
        row = matches[0]
        return int(row["requery_index"]), row

    def propagate_if_selected(
        self,
        *,
        selected_query_name: str | None = None,
        selected_candidate_uid: str | None = None,
        selection_audit: Mapping[str, Any] | None = None,
        none_score: float | None = None,
        margin: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rerun exactly one selected query and propagate it to the frozen horizon."""

        if not self._probe_candidates and not self._query_audits:
            raise RuntimeError("query_current_frame() must precede propagation")
        if selection_audit is not None:
            _reject_future_or_gt_metadata(selection_audit, "selection_audit")
        if selected_query_name is None and selected_candidate_uid is None:
            self._selection_audit = {
                "status": "NONE",
                "selected_query_name": None,
                "selected_candidate_uid": None,
                "none_score": None if none_score is None else float(none_score),
                "margin": None if margin is None else float(margin),
                "selector": None if selection_audit is None else deepcopy(dict(selection_audit)),
                "runtime_future_gt_used": False,
            }
            self.close()
            return []
        query_index, probe_row = self._find_query(selected_query_name, selected_candidate_uid)
        query_name = str(probe_row["requery_name"])
        spec = next(spec for spec in self.query_specs if str(spec["name"]) == query_name)
        prompt_box = query_box(self._predicted_box or [], spec)
        self._selected_query_name = query_name
        self._selected_candidate_uid = str(probe_row["candidate_uid"])
        self._selection_audit = {
            "status": "SELECTED",
            "selected_query_name": query_name,
            "selected_candidate_uid": str(probe_row["candidate_uid"]),
            "none_score": None if none_score is None else float(none_score),
            "margin": None if margin is None else float(margin),
            "selector": None if selection_audit is None else deepcopy(dict(selection_audit)),
            "probe_then_rerun_active_session": True,
            "runtime_future_gt_used": False,
        }
        backend: Any | None = None
        session: TargetScopedCorrectionSession | None = None
        try:
            backend, session = self._new_session(query_name)
            self._active_backend = backend
            self._active_session = session
            session.seed_from_human_box(prompt_box)
            outputs = session.propagate_to(self.end_frame)
            self._future_candidates = []
            self._future_frame_coverage = []
            for frame in range(self.trigger_frame, self.end_frame + 1):
                observations = list(outputs.get(frame, []))
                frame_rows = [
                    row
                    for row in (
                        self._serialize_propagated_observation(
                            observation,
                            frame=frame,
                            query_index=query_index,
                            query_name=query_name,
                            query_box_xyxy=prompt_box,
                        )
                        for observation in observations
                    )
                    if row is not None
                ]
                for row in frame_rows:
                    row["source_session_id"] = session.session_id
                    row["target_session_scope"] = session.target_session_scope
                    row["native_scope"] = session.target_session_scope
                    row["native_tid_scope"] = session.target_session_scope
                    row["probe_candidate_uid"] = str(probe_row["candidate_uid"])
                self._future_candidates.extend(frame_rows)
                self._future_frame_coverage.append(
                    {
                        "global_frame": int(frame),
                        "local_frame": int(frame - self.trigger_frame),
                        "candidate_count": len(frame_rows),
                        "candidate_uids": [str(row["candidate_uid"]) for row in frame_rows],
                        "runtime_future_gt_used": False,
                    }
                )
            self._active_session_audit = session.audit()
            if not any(int(row["frame"]) == self.trigger_frame for row in self._future_candidates):
                raise RuntimeError("selected active session did not expose a trigger-frame candidate")
            self._selection_audit["selected_active_candidate_uid"] = next(
                str(row["candidate_uid"])
                for row in self._future_candidates
                if int(row["frame"]) == self.trigger_frame
            )
            return deepcopy(self._future_candidates)
        except Exception:
            raise

    def _raw_from_causal_state(self) -> int | None:
        for key in ("previous_raw_sam_id", "last_raw_sam_id", "current_raw_sam_id", "raw_sam_id"):
            value = self._causal_state.get(key)
            if value is not None:
                return int(value)
        return None

    def audit(self) -> dict[str, Any]:
        selection = deepcopy(self._selection_audit)
        selected_row = next(
            (
                row
                for row in self._future_candidates
                if str(row.get("probe_candidate_uid")) == self._selected_candidate_uid
                or str(row["candidate_uid"]) == self._selected_candidate_uid
            ),
            None,
        )
        new_raw = None if selected_row is None else int(selected_row["official_raw_sam_id"])
        old_raw = self._raw_from_causal_state()
        status = "CLOSED_NONE" if selection and selection.get("status") == "NONE" else (
            "PASS_SELECTED" if selected_row is not None else "STARTED"
        )
        return {
            "schema_version": "N72R10_FUTURE_FRAME_REQUERY_SESSION_AUDIT_V1",
            "status": status,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_frame": int(self.event_frame),
            "trigger_frame": self._trigger_frame,
            "end_frame": self._end_frame,
            "local_frame_zero_global": self._trigger_frame,
            "window_mapping": deepcopy(self._window_mapping),
            "predicted_box_xyxy": deepcopy(self._predicted_box),
            "causal_state": deepcopy(self._causal_state),
            "query_specs": [deepcopy(dict(spec)) for spec in self.query_specs],
            "query_audits": deepcopy(self._query_audits),
            "probe_candidate_count": len(self._probe_candidates),
            "probe_candidates": deepcopy(self._probe_candidates),
            "selected_query_name": self._selected_query_name,
            "selected_candidate_uid": self._selected_candidate_uid,
            "future_candidate_count": len(self._future_candidates),
            "future_frame_coverage": deepcopy(self._future_frame_coverage),
            "invalid_observation_audit": deepcopy(self._invalid_observation_audit),
            "active_session_audit": deepcopy(self._active_session_audit),
            "selection": selection,
            "raw_rebinding": {
                "old_raw_sam_id": old_raw,
                "old_source": "causal_runtime_state",
                "new_raw_sam_id": new_raw,
                "new_source": FUTURE_FRAME_REQUERY,
                "public_id": int(self.target_public_id),
                "public_id_changed": False,
                "association_state_unchanged": True,
                "lineage_unchanged": True,
                "requery_session_id": None
                if self._active_session_audit is None
                else self._active_session_audit.get("session_id"),
            },
            "event_frame_memory_read": False,
            "first_memory_visible_frame": None if self._trigger_frame is None else int(self._trigger_frame + 1),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "closed": bool(self._closed),
        }

    def close(self) -> None:
        if self._closed:
            return
        session = self._active_session
        backend = self._active_backend
        self._active_session = None
        self._active_backend = None
        self._cleanup_backend_session(session, backend)
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._window_path = None
        self._closed = True
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "FUTURE_FRAME_REQUERY",
    "FUTURE_FRAME_REQUERY_CANDIDATE_KIND",
    "FutureFrameRequerySession",
    "QUERY_SPECS",
    "query_box",
]
