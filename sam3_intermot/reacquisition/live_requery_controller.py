"""Causal on-demand future requery controller for N72R11.

The controller owns lifecycle only.  It never scores candidates, chooses a
public identity, consults GT, or infers an ID from a native SAM object.  A
caller supplies the causal uncertainty trigger and the already computed
assignment-aware selection result; this module then probes the frozen
``FutureFrameRequerySession`` and retains only the explicitly selected
session's propagated evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from sam3_intermot.reacquisition.future_requery_session import FutureFrameRequerySession


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _close_and_release(value: Any) -> None:
    try:
        value.close()
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


@dataclass
class ActiveRequerySource:
    """The single selected rescue source currently allowed in a frame pool."""

    trigger_frame: int
    selected_candidate_uid: str
    selected_query_name: str
    session_audit: dict[str, Any]
    rows_by_frame: dict[int, list[dict[str, Any]]]
    source_sha256: str | None


class LiveFutureRequeryController:
    """Start and retain fresh future sessions only after causal uncertainty."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], Any],
        sequence: str,
        event_id: str,
        event_frame: int,
        target_public_id: int,
        frame_paths: Sequence[Any] | Mapping[int, Any],
        feature_fn: Callable[[int, Sequence[float]], Any] | None,
        end_frame: int,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if not callable(feature_fn) and feature_fn is not None:
            raise TypeError("feature_fn must be callable or None")
        if int(end_frame) < int(event_frame) + 1:
            raise ValueError("end_frame must include at least event_frame+1")
        self.backend_factory = backend_factory
        self.sequence = str(sequence)
        self.event_id = str(event_id)
        self.event_frame = int(event_frame)
        self.target_public_id = int(target_public_id)
        self.frame_paths = frame_paths
        self.feature_fn = feature_fn
        self.end_frame = int(end_frame)
        self.active_source: ActiveRequerySource | None = None
        self.last_trigger_frame: int | None = None
        self.trigger_count = 0
        self.probe_count = 0
        self.selected_count = 0
        self.requery_sessions_started = 0
        self.closed = False
        self._pending_session: FutureFrameRequerySession | None = None
        self._pending_probe_rows: list[dict[str, Any]] = []
        self._pending_start_audit: dict[str, Any] | None = None
        self._retired_source_count = 0

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("live future requery controller is closed")

    def _retire_active_source(self) -> None:
        if self._pending_session is not None:
            _close_and_release(self._pending_session)
            self._pending_session = None
            self._pending_probe_rows = []
            self._pending_start_audit = None
        if self.active_source is not None:
            self._retired_source_count += 1
            self.active_source = None
        gc.collect()

    def probe(
        self,
        *,
        frame: int,
        predicted_box: Sequence[float],
        causal_state: Mapping[str, Any],
    ) -> tuple[FutureFrameRequerySession, list[dict[str, Any]]]:
        """Start one fresh session and return its current-frame probe rows.

        No propagation happens here.  The returned session is committed or
        explicitly rejected by :meth:`commit`.
        """

        self._ensure_open()
        trigger_frame = int(frame)
        if trigger_frame <= self.event_frame or trigger_frame > self.end_frame:
            raise ValueError("probe frame must be in event_frame+1..end_frame")
        if self.last_trigger_frame is not None and trigger_frame <= self.last_trigger_frame:
            raise ValueError("probe frames must be strictly increasing")
        self._retire_active_source()
        session = FutureFrameRequerySession(
            backend_factory=self.backend_factory,
            sequence=self.sequence,
            event_id=self.event_id,
            event_frame=self.event_frame,
            target_public_id=self.target_public_id,
            frame_paths=self.frame_paths,
            feature_fn=self.feature_fn,
        )
        self.trigger_count += 1
        self.requery_sessions_started += 1
        self.last_trigger_frame = trigger_frame
        try:
            self._pending_start_audit = session.start_from_frame(
                trigger_frame,
                predicted_box,
                causal_state,
                end_frame=self.end_frame,
                main_y_pre_frozen=True,
            )
            rows = session.query_current_frame()
            self._pending_session = session
            self._pending_probe_rows = deepcopy(rows)
            self.probe_count += len(rows)
            return session, deepcopy(rows)
        except Exception:
            _close_and_release(session)
            raise

    @staticmethod
    def _rows_by_frame(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in rows:
            row = deepcopy(dict(item))
            frame = int(row.get("frame", -1))
            if frame < 0:
                raise ValueError("selected future candidate lacks a valid frame")
            if row.get("public_id") is not None or row.get("public_id_inference") is not False:
                raise ValueError("future rescue candidate carries public-ID authority")
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
                raise ValueError("future rescue candidate violates runtime GT boundary")
            grouped.setdefault(frame, []).append(row)
        for frame, frame_rows in grouped.items():
            uids = [str(row.get("candidate_uid")) for row in frame_rows]
            if any(uid in {"", "None"} for uid in uids) or len(uids) != len(set(uids)):
                raise ValueError(f"selected future rows contain duplicate/missing UIDs at frame {frame}")
        return grouped

    def commit(
        self,
        *,
        session: FutureFrameRequerySession,
        selected_candidate_uid: str | None,
        selection_audit: Mapping[str, Any],
        none_score: float | None,
        margin: float | None,
    ) -> list[dict[str, Any]]:
        """Commit exactly one probed query or close it as an explicit NONE."""

        self._ensure_open()
        if session is not self._pending_session:
            raise ValueError("commit session is not the controller's pending probe session")
        probe_uids = {str(row["candidate_uid"]): str(row.get("requery_name")) for row in self._pending_probe_rows}
        if selected_candidate_uid is None:
            session.propagate_if_selected(
                selection_audit=selection_audit,
                none_score=none_score,
                margin=margin,
            )
            if not session.audit().get("closed", False):
                session.close()
            self._pending_session = None
            self._pending_probe_rows = []
            self._pending_start_audit = None
            gc.collect()
            return []
        selected_uid = str(selected_candidate_uid)
        if selected_uid not in probe_uids:
            raise ValueError("selected_candidate_uid must identify one successful probe row")
        selected_query_name = probe_uids[selected_uid]
        if not selected_query_name:
            raise ValueError("selected probe row has no query name")
        rows = session.propagate_if_selected(
            selected_query_name=selected_query_name,
            selected_candidate_uid=selected_uid,
            selection_audit=selection_audit,
            none_score=none_score,
            margin=margin,
        )
        grouped = self._rows_by_frame(rows)
        session_audit = session.audit()
        active = ActiveRequerySource(
            trigger_frame=int(session.trigger_frame),
            selected_candidate_uid=selected_uid,
            selected_query_name=selected_query_name,
            session_audit=session_audit,
            rows_by_frame=grouped,
            source_sha256=_digest({str(frame): frame_rows for frame, frame_rows in sorted(grouped.items())}),
        )
        self.active_source = active
        self.selected_count += 1
        # ``propagate_if_selected`` has fully materialized the selected rows and
        # the audit above is a deep-copied observation of the completed
        # propagation.  Keeping the FutureFrameRequerySession alive here would
        # retain its official backend, session state, and frame window while
        # the controller only consumes ``rows_by_frame``.  Release that
        # resource before returning; ActiveRequerySource is intentionally an
        # observation-only record and never owns an executable session.
        _close_and_release(session)
        session_audit["closed_after_materialization"] = True
        active.session_audit = session_audit
        self._pending_session = None
        self._pending_probe_rows = []
        self._pending_start_audit = None
        gc.collect()
        return deepcopy(rows)

    def active_candidates(self, frame: int) -> list[dict[str, Any]]:
        """Return only the currently selected live rescue rows for a frame."""

        self._ensure_open()
        if self.active_source is None:
            return []
        return deepcopy(self.active_source.rows_by_frame.get(int(frame), []))

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R11_LIVE_FUTURE_REQUERY_CONTROLLER_AUDIT_V1",
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_frame": self.event_frame,
            "end_frame": self.end_frame,
            "last_trigger_frame": self.last_trigger_frame,
            "trigger_count": self.trigger_count,
            "probe_count": self.probe_count,
            "selected_count": self.selected_count,
            "requery_sessions_started": self.requery_sessions_started,
            "retired_source_count": self._retired_source_count,
            "pending_probe_count": len(self._pending_probe_rows),
            "active_source": None if self.active_source is None else {
                "trigger_frame": self.active_source.trigger_frame,
                "selected_candidate_uid": self.active_source.selected_candidate_uid,
                "selected_query_name": self.active_source.selected_query_name,
                "source_sha256": self.active_source.source_sha256,
                "frames": sorted(self.active_source.rows_by_frame),
                "session_audit": deepcopy(self.active_source.session_audit),
            },
            "event_frame_memory_read": False,
            "first_memory_visible_frame": self.event_frame + 1,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "closed": self.closed,
        }

    def close(self) -> None:
        if self.closed:
            return
        self._retire_active_source()
        self.closed = True
        self.backend_factory = None  # type: ignore[assignment]
        self.frame_paths = ()
        self.feature_fn = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["ActiveRequerySource", "LiveFutureRequeryController"]
