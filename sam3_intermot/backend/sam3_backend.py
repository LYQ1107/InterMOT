"""Real SAM 3.1 backend.

This module wraps the official ``sam3`` package (pinned commit
``4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b``) and the SHA256-verified public
mirror checkpoint (``AEmotionStudio/sam3.1``).  See ``sam3_compat.py`` for the
documented torch-2.5 compatibility shim.

The official multiplex ``add_prompt`` API resets the session on every prompt,
so incremental single-object calls are implemented by re-prompting the full
set of active boxes on the requested frame.  This is a stable adapter-level
behaviour; no third-party source file is modified.
"""

import os
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam3_intermot.backend import sam3_compat  # noqa: F401  (installs shims)
from sam3_intermot.backend.base import NotSupportedError, PromptVideoTrackerBackend
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.observations.mask_to_box import mask_to_box
from sam3_intermot.observations.observation import box_xyxy_to_xywh


class CheckpointUnavailableError(RuntimeError):
    """Raised when the SAM 3.1 checkpoint cannot be loaded."""


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Sam3Backend(PromptVideoTrackerBackend):
    """Adapter around ``build_sam3_multiplex_video_predictor``."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        use_fa3: bool = False,
        use_rope_real: bool = True,
        compile: bool = False,
        warm_up: bool = False,
        session_expiration_sec: int = 1200,
        output_prob_thresh: float = 0.5,
        async_loading_frames: bool = False,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.max_num_objects = max_num_objects
        self.multiplex_count = multiplex_count
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real
        self.compile = compile
        self.warm_up = warm_up
        self.session_expiration_sec = session_expiration_sec
        self.output_prob_thresh = output_prob_thresh
        self.async_loading_frames = async_loading_frames
        self.device = device
        self._predictor = None
        self._session_id: Optional[str] = None
        self._frame_h = 0
        self._frame_w = 0
        self._output_cache: Dict[int, List[PromptObjectObservation]] = {}
        self._objects: Dict[int, dict] = {}
        self._ext_to_sam: Dict[int, int] = {}
        self._sam_to_ext: Dict[int, int] = {}
        self._last_prompt_frame: Optional[int] = None
        self._text_prompt: Optional[str] = None
        self._prompt_fallback_log: List[dict] = []
        self._resume_repair_log: List[dict] = []

    # ------------------------------------------------------------------
    def _require_checkpoint(self) -> None:
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise CheckpointUnavailableError(
                "SAM 3.1 checkpoint (sam3.1_multiplex.pt) is unavailable; "
                "expected at config backend.checkpoint_path."
            )

    def _ensure_model(self) -> None:
        self._require_checkpoint()
        if self._predictor is not None:
            return
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "sam3 package is not importable; install the pinned official repo first"
            ) from exc
        self._predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=self.checkpoint_path,
            max_num_objects=self.max_num_objects,
            multiplex_count=self.multiplex_count,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
            compile=self.compile,
            warm_up=self.warm_up,
            session_expiration_sec=self.session_expiration_sec,
            default_output_prob_thresh=self.output_prob_thresh,
            async_loading_frames=self.async_loading_frames,
        )
        # The official builder enables a 15-frame demo hotstart buffer and
        # masklet confirmation, which withholds per-frame outputs for the first
        # 15 frames.  For MOT we need immediate per-frame outputs, so these
        # runtime knobs are switched off at the adapter level (no third-party
        # source file is modified).
        if hasattr(self._predictor.model, "hotstart_delay"):
            self._predictor.model.hotstart_delay = 0
        if hasattr(self._predictor.model, "masklet_confirmation_enable"):
            self._predictor.model.masklet_confirmation_enable = False
        tracker = getattr(self._predictor.model, "tracker", None)
        tracker_models = [tracker]
        if tracker is not None and hasattr(tracker, "model"):
            tracker_models.append(tracker.model)
        for tm in tracker_models:
            if tm is None:
                continue
            # Long-sequence MOT keeps per-frame outputs and memories on GPU by
            # default, which exhausts a 40GB card after a few hundred frames.
            # Offloading outputs to CPU and trimming past non-conditioning
            # memory is a runtime adapter setting (no source modification).
            if hasattr(tm, "offload_output_to_cpu_for_eval"):
                tm.offload_output_to_cpu_for_eval = True

        self._runtime_memory_policy = {
            "offload_video_to_cpu": True,
            "offload_output_to_cpu_for_eval": True,
            # The pinned outer multiplex init_state does not accept the
            # official offload_state_to_cpu keyword.  N35's adapter-local
            # nested partial reached CPU/CUDA mismatches in both
            # reconditioning and new-object addition, so N36 intentionally
            # uses process isolation instead of an unsupported mixed-device
            # state.  Keep this false and explicit in every tape artifact.
            "offload_state_to_cpu": False,
            "trim_past_non_cond_mem_for_eval": False,
            "recondition_every_nth_frame": int(
                getattr(self._predictor.model, "recondition_every_nth_frame", 16)
            ),
            "use_iom_recondition": bool(
                getattr(self._predictor.model, "use_iom_recondition", True)
            ),
            "source": "official_video_cpu_offload_plus_adapter_output_offload_process_isolation",
        }

    def _require_session(self) -> None:
        self._ensure_model()
        if self._session_id is None:
            raise RuntimeError("start_video must be called before any prompt")

    # ------------------------------------------------------------------
    def start_video(self, video_source: str) -> str:
        self._ensure_model()
        # Official Sam3BasePredictor.start_session unconditionally passes
        # offload_state_to_cpu to model.init_state, but
        # Sam3MultiplexTrackingWithInteractivity.init_state does not accept it
        # at pinned commit 4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b.  We
        # therefore call the official model.init_state directly and register
        # the session in the predictor's session table (adapter workaround;
        # no third-party source file is modified).
        inference_state = self._predictor.model.init_state(
            resource_path=video_source,
            offload_video_to_cpu=True,
            async_loading_frames=self.async_loading_frames,
        )
        session_id = str(uuid.uuid4())
        self._predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        self._session_id = session_id
        if os.path.isdir(video_source):
            from PIL import Image

            images = sorted(
                (
                    p
                    for p in os.scandir(video_source)
                    if p.name.lower().endswith((".jpg", ".jpeg", ".png"))
                ),
                key=lambda p: p.name,
            )
            if images:
                img = Image.open(images[0].path)
                self._frame_w, self._frame_h = img.size
        self._output_cache = {}
        self._objects = {}
        self._ext_to_sam = {}
        self._sam_to_ext = {}
        self._last_prompt_frame = None
        self._text_prompt = None
        self._prompt_fallback_log = []
        self._resume_repair_log = []
        return self._session_id

    # ------------------------------------------------------------------
    def detect_concept(
        self, frame_idx: int, text_prompt: str
    ) -> List[PromptObjectObservation]:
        self._require_session()
        self._text_prompt = text_prompt
        boxes = [o["box"] for o in self._objects.values()]
        obs = self._send_prompt(
            frame_idx,
            text=text_prompt,
            boxes=boxes if boxes else None,
            source="concept_detection",
        )
        self._apply_stable_ids(obs)
        return obs

    def add_box(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: np.ndarray,
    ) -> PromptObjectObservation:
        self._require_session()
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        if box.size != 4:
            raise ValueError("box_xyxy must have exactly 4 elements")
        human_box = self._clip_box(box)
        is_new = object_id not in self._objects
        source = "human_add" if is_new else "human_correction"
        if is_new:
            self._objects[object_id] = {
                "box": human_box.copy(),
                "human_box": human_box.copy(),
                "frame": int(frame_idx),
                "source": source,
            }
        prompt_box = self._prompt_with_variants(
            int(frame_idx), object_id, human_box, source
        )
        self._objects[object_id]["box"] = prompt_box.copy()
        self._objects[object_id]["human_box"] = human_box.copy()
        self._objects[object_id]["frame"] = int(frame_idx)
        self._last_prompt_frame = int(frame_idx)
        obs = self._human_observation(int(frame_idx), object_id, human_box, source)
        self._add_human_to_cache(int(frame_idx), obs)
        return obs

    def seed_box_from_past_state(
        self,
        frame_idx: int,
        object_id: int,
        box_xyxy: np.ndarray,
    ) -> Optional[PromptObjectObservation]:
        """Re-initialize an independent session from a past-state box.

        The box must come from a previously frozen same-sequence runtime
        observation.  It is not a human event, GT, or a future-frame input.
        The official SAM3 prompt path is retained, while the returned
        observation is explicitly marked as ``past_state_seed`` and not human
        verified.  Cross-session public identity is assigned only later by
        the N72R2 handover ledger; ``object_id`` is an adapter-local key.
        """

        self._require_session()
        seed_id = int(object_id)
        if seed_id in self._objects:
            raise ValueError(f"past-state seed object already exists: {seed_id}")
        box = self._clip_box(np.asarray(box_xyxy, dtype=float).reshape(-1))
        self._objects[seed_id] = {
            "box": box.copy(),
            "human_box": box.copy(),
            "frame": int(frame_idx),
            "source": "past_state_seed",
        }
        prompt_box = self._prompt_with_variants(
            int(frame_idx), seed_id, box, "past_state_seed"
        )
        self._objects[seed_id]["box"] = prompt_box.copy()
        self._objects[seed_id]["human_box"] = box.copy()
        self._objects[seed_id]["frame"] = int(frame_idx)
        self._last_prompt_frame = int(frame_idx)
        target = self._find_obs_for_ext(
            self._output_cache.get(int(frame_idx), []), seed_id
        )
        if target is None:
            # Do not leave an unobservable object in subsequent whole-set
            # official prompts.  The caller records this as candidate-recall
            # evidence instead of upgrading it to an authority binding.
            self._objects.pop(seed_id, None)
            self._ext_to_sam.pop(seed_id, None)
            for mapped_sam, mapped_external in list(self._sam_to_ext.items()):
                if int(mapped_external) == seed_id:
                    self._sam_to_ext.pop(mapped_sam, None)
            return None
        result = target.copy()
        result.source = "past_state_seed"
        result.is_human_verified = False
        return result

    def rebind_past_state_boxes(
        self,
        frame_idx: int,
        seeds: Sequence[Tuple[int, np.ndarray]],
    ) -> dict:
        """Re-initialize one independent session with one official multi-box prompt.

        The pinned multiplex model accepts a complete ``bounding_boxes`` batch,
        and resets/rebuilds its prompt state on every ``add_prompt`` request.
        Calling that real interface once for the whole persisted object set is
        materially different from repeatedly adding one seed (each repeated
        call can discard the preceding seed).  This adapter-level primitive is
        deliberately limited to same-sequence past runtime boxes: it does not
        read GT, assign public IDs, or infer a match from numeric IDs.

        The returned mapping contains only observations that the official
        response exposed in a one-to-one box match.  Unobserved or ambiguous
        seeds remain explicit failures; they are never converted into a
        TrackManager/public-authority binding.
        """

        self._require_session()
        normalized: list[tuple[int, np.ndarray]] = []
        seen: set[int] = set()
        for object_id, box_xyxy in seeds:
            external_id = int(object_id)
            if external_id in seen or external_id in self._objects:
                raise ValueError(f"duplicate or existing past-state object id: {external_id}")
            raw_box = np.asarray(box_xyxy, dtype=float).reshape(-1)
            if raw_box.size != 4 or not np.all(np.isfinite(raw_box)):
                raise ValueError(f"invalid past-state box for object {external_id}")
            box = self._clip_box(raw_box)
            seen.add(external_id)
            normalized.append((external_id, box))
        if not normalized:
            raise ValueError("at least one past-state box is required")

        self._objects = {
            external_id: {
                "box": box.copy(),
                "human_box": box.copy(),
                "frame": int(frame_idx),
                "source": "past_state_rebind",
            }
            for external_id, box in normalized
        }
        self._ext_to_sam.clear()
        self._sam_to_ext.clear()
        self._last_prompt_frame = int(frame_idx)

        prompt_boxes = [box.copy() for _, box in normalized]
        attempts: list[dict] = []
        obs_list = self._send_prompt(
            int(frame_idx), boxes=prompt_boxes, source="past_state_rebind"
        )
        self._apply_stable_ids(obs_list)
        attempts.append(
            {
                "variant": "persisted_box",
                "requested_count": len(normalized),
                "observed_count": len(obs_list),
            }
        )

        # The regular adapter already uses this official, deterministic
        # centered shrink for oversized boxes.  Apply it to the *whole* batch
        # only if the first complete prompt did not expose every seed.
        if len(obs_list) < len(normalized):
            sanitized = [(external_id, self._sanitize_box(box)) for external_id, box in normalized]
            for external_id, box in sanitized:
                self._objects[external_id]["box"] = box.copy()
            obs_list = self._send_prompt(
                int(frame_idx),
                boxes=[box.copy() for _, box in sanitized],
                source="past_state_rebind_sanitized",
            )
            self._apply_stable_ids(obs_list)
            attempts.append(
                {
                    "variant": "sanitized_box",
                    "requested_count": len(sanitized),
                    "observed_count": len(obs_list),
                }
            )

        # _match_outputs_to_ext performs one-to-one greedy IoU matching.  Use
        # the established map only when an exposed raw observation exists;
        # duplicate boxes cannot silently satisfy two persisted objects.
        raw_by_id = {
            int(obs.raw_sam_object_id): obs
            for obs in obs_list
            if obs.raw_sam_object_id is not None
        }
        recovered: dict[int, PromptObjectObservation] = {}
        failures: list[dict] = []
        for external_id, _ in normalized:
            sam_id = self._ext_to_sam.get(external_id)
            target = raw_by_id.get(int(sam_id)) if sam_id is not None else None
            if target is None:
                failures.append(
                    {
                        "object_id": external_id,
                        "reason": "official_multi_box_prompt_did_not_return_unique_observation",
                    }
                )
                self._objects.pop(external_id, None)
                old_sam = self._ext_to_sam.pop(external_id, None)
                if old_sam is not None:
                    self._sam_to_ext.pop(int(old_sam), None)
                continue
            result = target.copy()
            result.source = "past_state_rebind"
            result.is_human_verified = False
            recovered[external_id] = result

        return {
            "frame_idx": int(frame_idx),
            "requested_count": len(normalized),
            "recovered_count": len(recovered),
            "failure_count": len(failures),
            "failures": failures,
            "attempts": attempts,
            "one_to_one_matching": True,
            "source": "official_multiplex_add_prompt_bounding_boxes",
            "runtime_future_gt_used": False,
            "observations": recovered,
        }

    def add_points(
        self,
        frame_idx: int,
        object_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> PromptObjectObservation:
        # Official multiplex add_prompt accepts text/boxes only, not points.
        raise NotSupportedError(
            "official SAM 3.1 multiplex handle_request does not expose point prompts; "
            "convert points to a box with observations.mask_to_box and use add_box"
        )

    def add_mask(
        self,
        frame_idx: int,
        object_id: int,
        mask: np.ndarray,
    ) -> PromptObjectObservation:
        # The official handle_request API exposed by SAM 3.1 multiplex accepts
        # text/boxes but not masks.  Do not fake native mask support.
        if mask is None:
            raise NotSupportedError(
                "official SAM 3.1 multiplex handle_request does not expose mask "
                "prompts; pass a real 2D mask to convert it to a box"
            )
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
        *,
        allow_prompt_fallback: bool = True,
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
            raise NotSupportedError(
                "official SAM 3.1 multiplex handle_request does not expose point prompts"
            )
        else:
            raise ValueError("correct_object requires box, points or mask")
        human_box = self._clip_box(box)
        self._objects[object_id]["human_box"] = human_box.copy()
        if allow_prompt_fallback:
            prompt_box = self._prompt_with_variants(
                int(frame_idx), object_id, human_box, "human_correction"
            )
        else:
            # N31 uses this branch to measure the causal effect of the
            # unmodified official prompt.  It deliberately makes one request
            # and does not silently shrink/retry an oversized box.
            self._objects[object_id]["box"] = human_box.copy()
            obs = self._send_prompt(
                int(frame_idx),
                boxes=[o["box"] for o in self._objects.values()],
                source="human_correction_no_fallback",
            )
            self._find_obs_for_ext(obs, object_id)
            self._apply_stable_ids(obs)
            prompt_box = human_box.copy()
        self._objects[object_id]["box"] = prompt_box.copy()
        self._objects[object_id]["frame"] = int(frame_idx)
        self._last_prompt_frame = int(frame_idx)
        obs = self._human_observation(
            int(frame_idx), object_id, human_box, "human_correction"
        )
        self._add_human_to_cache(int(frame_idx), obs)
        return obs

    def propagate(
        self,
        start_frame: int,
        end_frame: int,
        start_frame_index: Optional[int] = None,
        *,
        keep_masks: bool = True,
        cache_outputs: bool = True,
        output_callback: Optional[
            Callable[[int, List[PromptObjectObservation]], None]
        ] = None,
    ) -> dict:
        self._require_session()
        if not self._objects and self._text_prompt is None:
            return {}
        # Existing callers retain the historical in-memory return/cache
        # behavior.  Long candidate-tape exports can opt into a streaming
        # callback so parsed masks are written one frame at a time instead of
        # accumulating the whole video in the adapter.
        outputs: dict = {}
        req = dict(
            type="propagate_in_video",
            session_id=self._session_id,
            propagation_direction="forward",
        )
        start_idx = start_frame_index
        if start_idx is None:
            start_idx = self._last_prompt_frame if self._last_prompt_frame is not None else 0
        req["start_frame_index"] = int(start_idx)
        self._prepare_resumable_official_stream(int(start_idx), int(end_frame))
        # Intentionally do NOT pass max_frame_num_to_track.  The pinned official
        # code computes a detector feature window of length
        # max_frame_num_to_track, while its propagation order is inclusive and
        # therefore contains max+1 frames; passing a small max makes the final
        # frame of the window have zero feature batch size.  Omitting it lets
        # the official default (track to end of video) drive the window, and the
        # caller can stop consuming responses after end_frame.
        # ``handle_stream_request`` is an inference-mode generator.  Closing
        # it explicitly is required when the adapter stops at an early
        # ``end_frame``; otherwise its inference-mode context can remain live
        # while the caller creates the differentiable LoRA state.  Such a
        # state reports ``requires_grad=True`` but produces no autograd graph.
        stream = self._predictor.handle_stream_request(request=req)
        try:
            for response in stream:
                frame_idx = int(response["frame_index"])
                if frame_idx > end_frame:
                    break
                if frame_idx < start_frame:
                    continue
                obs_list = self._parse_outputs(
                    response, frame_idx, "automatic_propagation"
                )
                if not keep_masks:
                    for obs in obs_list:
                        obs.mask = np.zeros((1, 1), dtype=bool)
                self._apply_stable_ids(obs_list)
                if cache_outputs:
                    self._output_cache[frame_idx] = obs_list
                if output_callback is not None:
                    output_callback(
                        frame_idx,
                        [observation.copy() for observation in obs_list],
                    )
                if cache_outputs or output_callback is None:
                    outputs[frame_idx] = obs_list
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return outputs

    def _prepare_resumable_official_stream(self, start_frame: int, end_frame: int) -> None:
        """Repair the pinned multiplex fetch/full-propagation transition.

        The official interactivity planner treats a second propagation call as
        a cache fetch.  If the first call was intentionally stopped before a
        detector batch completed, that cache can end at a batch boundary and
        the fetch raises ``KeyError`` for the first uncached frame.  Asking the
        official planner to replay its previous propagation action keeps the
        already materialized tracker state and computes the missing suffix;
        this is an adapter-level continuation wrapper, not a third-party edit.
        """

        if start_frame <= 0 or self._predictor is None or self._session_id is None:
            return
        entry = self._predictor._all_inference_states.get(self._session_id)
        state = entry.get("state") if isinstance(entry, dict) else None
        if not isinstance(state, dict):
            return
        cached = state.get("cached_frame_outputs", {})
        num_frames = int(state.get("num_frames", end_frame + 1))
        requested_end = min(int(end_frame), num_frames - 1)
        missing = [
            frame
            for frame in range(int(start_frame), requested_end + 1)
            if frame not in cached
        ]
        if not missing:
            return
        history = state.get("action_history", [])
        previous_action = history[-1] if history else None
        if not history or previous_action.get("type") == "propagation_cancel":
            return
        # A missing cache is normal before the first propagation after an
        # ``add``/``refine`` action.  In that state the official planner must
        # parse the user action itself; inserting a cancel would make the
        # action before cancel be ``add``/``refine`` and violate the planner's
        # propagation-type contract.  Cancel/replay is only a continuation
        # repair after a propagation stream was already started.
        if not str(previous_action.get("type", "")).startswith("propagation"):
            return
        model = getattr(self._predictor, "model", None)
        add_history = getattr(model, "add_action_history", None)
        if add_history is None:
            return
        add_history(state, action_type="propagation_cancel")
        self._resume_repair_log.append(
            {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "missing_cached_frame_count": len(missing),
                "first_missing_frame": int(missing[0]),
                "previous_action_type": str(previous_action.get("type")),
                "strategy": "official_action_history_cancel_then_replay",
            }
        )

    def remove_object(self, object_id: int) -> None:
        self._require_session()
        if object_id not in self._objects:
            raise ValueError(f"invalid object id: {object_id}")
        object_id = int(object_id)
        sam_id = self._ext_to_sam.get(object_id)
        del self._objects[object_id]
        self._ext_to_sam.pop(object_id, None)
        if sam_id is not None:
            self._sam_to_ext.pop(sam_id, None)
        # Cached observations may carry either the adapter-visible external
        # ID or the immutable raw official ID.  Remove both representations;
        # filtering only on ``sam_object_id == sam_id`` leaves stable-bound
        # observations behind.
        visible_ids = {object_id}
        raw_ids = {object_id}
        if sam_id is not None:
            visible_ids.add(int(sam_id))
            raw_ids.add(int(sam_id))
        for frame_obs in self._output_cache.values():
            frame_obs[:] = [
                o
                for o in frame_obs
                if int(o.sam_object_id) not in visible_ids
                and not (
                    o.raw_sam_object_id is not None
                    and int(o.raw_sam_object_id) in raw_ids
                )
            ]
        if self._objects and self._last_prompt_frame is not None:
            obs = self._send_prompt(
                self._last_prompt_frame,
                boxes=[o["box"] for o in self._objects.values()],
                source="automatic_propagation",
            )
            self._apply_stable_ids(obs)

    def reset_object(self, object_id: int) -> None:
        raise NotSupportedError(
            "official SAM 3.1 exposes whole-session reset, not per-object reset; "
            "the upper layer must re-add the object's initial prompt instead"
        )

    def reset_session(self) -> None:
        """Reset the current official session (Sam3BasePredictor.reset_session).

        This is the official segment-restart primitive: model state returns to
        its initialized form, then the caller re-adds prompts with add_box.
        """
        if self._predictor is None or self._session_id is None:
            raise RuntimeError("no active session to reset")
        self._predictor.handle_request(
            request=dict(type="reset_session", session_id=self._session_id)
        )
        self._output_cache.clear()
        self._objects.clear()
        self._ext_to_sam.clear()
        self._sam_to_ext.clear()
        self._last_prompt_frame = None
        self._resume_repair_log = []

    def get_frame_outputs(self, frame_idx: int) -> List[PromptObjectObservation]:
        return [o.copy() for o in self._output_cache.get(frame_idx, [])]

    def runtime_memory_policy(self) -> dict:
        """Return the actual adapter/runtime memory settings for provenance."""
        return dict(getattr(self, "_runtime_memory_policy", {}))

    def export_frame_candidates(
        self,
        frame_idx: int,
        *,
        embeddings: Optional[Sequence[np.ndarray]] = None,
        embedding_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        include_masks: bool = True,
        include_raw_provenance: bool = False,
    ) -> List[dict]:
        """Export all postprocessed SAM3 candidates cached for one frame.

        The official response has native object ids, boxes, masks and scores,
        but no public identity or embedding field.  This adapter method never
        drops a candidate: when no embedding is exposed (or a supplied
        machine ROI extractor fails), the row remains present with an
        explicit ``embedding_status``.  A caller may supply independent
        machine box-crop embeddings through ``embeddings`` or
        ``embedding_fn``; human evidence is not accepted by this method.
        """
        observations = self.get_frame_outputs(int(frame_idx))
        if embeddings is not None and len(embeddings) != len(observations):
            raise ValueError(
                "embeddings length must equal the complete cached candidate count"
            )
        rows: List[dict] = []
        for index, observation in enumerate(observations):
            embedding = None
            embedding_status = "NOT_EXPOSED"
            feature_source = "official_response_no_embedding"
            if embeddings is not None:
                embedding = embeddings[index]
                embedding_status = "MACHINE_ROI_FALLBACK"
                feature_source = "machine_roi_fallback"
            elif embedding_fn is not None:
                try:
                    embedding = embedding_fn(
                        int(frame_idx), np.asarray(observation.box_xyxy, dtype=float).copy()
                    )
                    embedding_status = "MACHINE_ROI_FALLBACK"
                    feature_source = "machine_roi_fallback"
                except Exception as exc:  # preserve candidate and provenance
                    embedding_status = "MACHINE_ROI_FALLBACK_FAILED"
                    feature_source = f"machine_roi_fallback_error:{type(exc).__name__}"
            if embedding is not None:
                try:
                    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                    if vector.size == 0 or not np.all(np.isfinite(vector)):
                        raise ValueError("embedding is empty or non-finite")
                    norm = float(np.linalg.norm(vector))
                    if norm <= 1e-6:
                        raise ValueError("embedding norm is zero")
                    embedding = (vector / norm).astype(np.float32)
                except (TypeError, ValueError):
                    embedding = None
                    if embedding_status == "MACHINE_ROI_FALLBACK":
                        embedding_status = "MACHINE_ROI_FALLBACK_FAILED"
                        feature_source = "machine_roi_fallback_invalid"
            row = {
                    "frame_idx": int(observation.frame_idx),
                    "native_tid": int(observation.sam_object_id),
                    "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).copy(),
                    "mask": (
                        np.asarray(observation.mask, dtype=bool).copy()
                        if include_masks and observation.mask is not None
                        else None
                    ),
                    "confidence": float(observation.confidence),
                    "presence_score": (
                        None
                        if observation.presence_score is None
                        else float(observation.presence_score)
                    ),
                    "source": str(observation.source),
                    "embedding": embedding,
                    "embedding_status": embedding_status,
                    "feature_source": feature_source,
                    "is_human_verified": bool(observation.is_human_verified),
                    "candidate_index": int(index),
                    # ``native_tid`` is the adapter-visible stable ID after
                    # binding.  The immutable official axis is exposed
                    # separately as ``raw_native_id`` when requested.
                    "native_id_source": "adapter_visible_stable_id_after_binding",
                    "legacy_native_tid_semantics": "adapter_visible_stable_id_after_binding",
                }
            # This is an explicit opt-in extension.  The historical export
            # keeps its byte/schema behaviour, while N72 can distinguish the
            # immutable official raw axis from the adapter-visible ID after
            # stable-ID binding.
            if include_raw_provenance:
                raw_id = observation.raw_sam_object_id
                row.update(
                    {
                        "raw_native_id": None if raw_id is None else int(raw_id),
                        "raw_native_id_source": (
                            "official_out_obj_ids"
                            if raw_id is not None
                            else "UNAVAILABLE_NOT_OFFICIAL_OBSERVATION"
                        ),
                        "adapter_external_id": int(observation.sam_object_id),
                    }
                )
            rows.append(row)
        return rows

    def export_frame_candidates_v2(
        self,
        frame_idx: int,
        *,
        metadata: dict,
        segment_local_ids: Sequence[str],
        sequence_global_ids: Sequence[str],
        embeddings: Optional[Sequence[np.ndarray]] = None,
        embedding_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> List[dict]:
        """Export a mandatory, provenance-complete Candidate V2 frame.

        The local/global axes are explicit inputs from the same run's binding
        ledger.  This method intentionally refuses to derive them from raw or
        adapter IDs, and it never adds a public identity.  The legacy exporter
        remains available for compatibility and is used only to obtain the
        same normalized machine feature values/common fields.
        """
        from sam3_intermot.provenance.candidate_v2 import build_candidate_v2_row

        observations = self.get_frame_outputs(int(frame_idx))
        if len(segment_local_ids) != len(observations) or len(sequence_global_ids) != len(observations):
            raise ValueError("Candidate V2 local/global axis lengths must equal the complete candidate count")
        context = dict(metadata)
        context.setdefault("session_id", self._session_id)
        legacy = self.export_frame_candidates(
            int(frame_idx), embeddings=embeddings, embedding_fn=embedding_fn, include_masks=True
        )
        if len(legacy) != len(observations):
            raise RuntimeError("legacy/V2 candidate count changed during export")
        rows: List[dict] = []
        for index, (observation, legacy_row) in enumerate(zip(observations, legacy)):
            # ``export_frame_candidates`` already normalizes an explicitly
            # supplied embedding.  Feed the original caller value to the V2
            # builder as well so the two schema projections share one
            # canonical normalization pass instead of normalizing the
            # legacy projection a second time.  The fallback path remains
            # compatible with embedding_fn and still preserves its audit
            # provenance.
            feature_input = (
                embeddings[index]
                if embeddings is not None
                else legacy_row.get("embedding")
            )
            row = build_candidate_v2_row(
                observation,
                metadata=context,
                candidate_index=index,
                segment_local_id=str(segment_local_ids[index]),
                sequence_global_id=str(sequence_global_ids[index]),
                feature=feature_input,
                feature_status=(
                    "AVAILABLE" if legacy_row.get("embedding") is not None else legacy_row.get("embedding_status")
                ),
                feature_source=str(legacy_row.get("feature_source", "official_response_no_embedding")),
            )
            rows.append(row)
        return rows

    def close(self) -> None:
        if self._predictor is not None and self._session_id is not None:
            try:
                self._predictor.handle_request(
                    request=dict(type="close_session", session_id=self._session_id)
                )
            except Exception:
                pass
        self._session_id = None
        self._output_cache.clear()
        self._objects.clear()
        self._ext_to_sam.clear()
        self._sam_to_ext.clear()
        self._last_prompt_frame = None
        self._text_prompt = None
        self._prompt_fallback_log = []
        self._resume_repair_log = []

    # ------------------------------------------------------------------
    def _bind_external_sam_id(self, external_id: int, sam_id: int) -> None:
        """Install one consistent public-ID/raw-SAM-ID binding.

        The official tracker is allowed to recreate a raw object namespace
        after a prompt or a low-level singleton write.  Keeping this update in
        one helper prevents stale inverse entries from routing later outputs
        to the wrong public identity.
        """

        external_id = int(external_id)
        sam_id = int(sam_id)

        old_sam = self._ext_to_sam.pop(external_id, None)
        if old_sam is not None and self._sam_to_ext.get(int(old_sam)) == external_id:
            self._sam_to_ext.pop(int(old_sam), None)

        old_external = self._sam_to_ext.pop(sam_id, None)
        if old_external is not None and self._ext_to_sam.get(int(old_external)) == sam_id:
            self._ext_to_sam.pop(int(old_external), None)

        # Also repair maps that were already inconsistent before this call.
        for mapped_external, mapped_sam in list(self._ext_to_sam.items()):
            if int(mapped_sam) == sam_id and int(mapped_external) != external_id:
                self._ext_to_sam.pop(mapped_external, None)
        for mapped_sam, mapped_external in list(self._sam_to_ext.items()):
            if int(mapped_external) == external_id and int(mapped_sam) != sam_id:
                self._sam_to_ext.pop(mapped_sam, None)

        self._ext_to_sam[external_id] = sam_id
        self._sam_to_ext[sam_id] = external_id

    def _send_prompt(
        self,
        frame_idx: int,
        *,
        boxes: Optional[List[np.ndarray]] = None,
        text: Optional[str] = None,
        source: str,
    ) -> List[PromptObjectObservation]:
        request: dict = dict(
            type="add_prompt",
            session_id=self._session_id,
            frame_index=int(frame_idx),
        )
        if text is not None:
            request["text"] = text
        if boxes is not None and len(boxes) > 0:
            request["bounding_boxes"] = [
                self._to_relative_xywh(np.asarray(b, dtype=float)) for b in boxes
            ]
            request["bounding_box_labels"] = [1] * len(boxes)
            request["clear_old_boxes"] = True
        response = self._predictor.handle_request(request=request)
        obs_list = self._parse_outputs(response, int(frame_idx), source)
        self._last_prompt_frame = int(frame_idx)
        self._output_cache[int(frame_idx)] = obs_list
        if boxes is not None and len(boxes) == len(self._objects):
            self._match_outputs_to_ext(obs_list, list(self._objects.values()))
        return obs_list

    def _match_outputs_to_ext(
        self,
        obs_list: List[PromptObjectObservation],
        prompt_objects: List[dict],
    ) -> None:
        """Associate model outputs with external prompt ids by box overlap."""
        used = set()
        for obs in sorted(obs_list, key=lambda o: -o.confidence):
            best_ext = None
            best_iou = 0.3
            for ext_id, obj in self._objects.items():
                if ext_id in used:
                    continue
                iou = _iou(obs.box_xyxy, obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ext = ext_id
            if best_ext is not None:
                used.add(best_ext)
                self._bind_external_sam_id(best_ext, obs.sam_object_id)

    def _find_obs_for_ext(
        self, obs_list: List[PromptObjectObservation], object_id: int
    ) -> Optional[PromptObjectObservation]:
        sam_id = self._ext_to_sam.get(object_id)
        for obs in obs_list:
            if obs.sam_object_id == sam_id:
                return obs
        # Fallback: match by IoU against the requested object's stored box.
        obj = self._objects.get(object_id)
        best = None
        best_iou = 0.0
        if obj is not None:
            for obs in obs_list:
                iou = _iou(obs.box_xyxy, obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best = obs
        if best is not None and best_iou > 0.3:
            self._bind_external_sam_id(object_id, best.sam_object_id)
            best.sam_object_id = object_id
            return best
        return None

    def _apply_stable_ids(
        self, obs_list: List[PromptObjectObservation]
    ) -> None:
        """Replace raw SAM model ids with stable external ids where known."""
        for obs in obs_list:
            raw_or_visible_id = (
                obs.raw_sam_object_id
                if obs.raw_sam_object_id is not None
                else obs.sam_object_id
            )
            ext = self._sam_to_ext.get(raw_or_visible_id)
            if ext is not None:
                obs.sam_object_id = ext

    # ------------------------------------------------------------------
    def _to_relative_xywh(self, box_xyxy: np.ndarray) -> list:
        x, y, w, h = box_xyxy_to_xywh(box_xyxy)
        return [
            float(x / self._frame_w),
            float(y / self._frame_h),
            float(w / self._frame_w),
            float(h / self._frame_h),
        ]

    def _sanitize_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        """Shrink oversized prompt boxes to the model's reliable prompt range.

        The pinned SAM 3.1 multiplex code returns no object for box prompts
        whose normalized width/height exceed roughly 0.07/0.28 of the frame
        (observed on 1920x1080 DanceTrack frames).  We keep the box centered
        and shrink only the dimensions that are too large, so user/GT boxes of
        any valid size remain usable.
        """
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        max_w = max(1.0, self._frame_w * 0.07)
        max_h = max(1.0, self._frame_h * 0.28)
        w = min(max(1.0, x2 - x1), max_w)
        h = min(max(1.0, y2 - y1), max_h)
        return np.asarray(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            dtype=float,
        )

    def _clip_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        """Clip a human-verified box to image bounds without changing its
        center or scale (except for pixels outside the frame)."""
        box = np.asarray(box_xyxy, dtype=float).reshape(-1)
        x1 = min(max(box[0], 0.0), float(self._frame_w))
        y1 = min(max(box[1], 0.0), float(self._frame_h))
        x2 = min(max(box[2], 0.0), float(self._frame_w))
        y2 = min(max(box[3], 0.0), float(self._frame_h))
        if x2 <= x1:
            x2 = min(x1 + 1.0, float(self._frame_w))
        if y2 <= y1:
            y2 = min(y1 + 1.0, float(self._frame_h))
        return np.asarray([x1, y1, x2, y2], dtype=float)

    def _human_prompt_variants(self, human_box: np.ndarray) -> List[np.ndarray]:
        """Prompt fallbacks for a human-verified box.

        The official multiplex API accepts boxes only (no point prompts), so
        fallbacks are: original clipped box, then a centered smaller box.  The
        current-frame MOT output always uses ``human_box``; the smaller box
        only affects SAM propagation from the next frame.
        """
        candidates = [human_box.copy(), self._sanitize_box(human_box).copy()]
        unique = []
        seen = set()
        for c in candidates:
            key = tuple(np.round(c, 2))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _prompt_with_variants(
        self,
        frame_idx: int,
        object_id: int,
        human_box: np.ndarray,
        source: str,
    ) -> np.ndarray:
        variants = self._human_prompt_variants(human_box)
        for vi, variant in enumerate(variants):
            self._objects[object_id]["box"] = variant.copy()
            obs = self._send_prompt(
                frame_idx,
                boxes=[o["box"] for o in self._objects.values()],
                source=source,
            )
            target = self._find_obs_for_ext(obs, object_id)
            if target is not None:
                if vi > 0:
                    self._prompt_fallback_log.append({
                        "object_id": object_id,
                        "frame_idx": frame_idx,
                        "variant_index": vi,
                        "human_box": human_box.tolist(),
                        "prompt_box": variant.tolist(),
                    })
                self._apply_stable_ids(obs)
                return variant
        self._prompt_fallback_log.append({
            "object_id": object_id,
            "frame_idx": frame_idx,
            "variant_index": -1,
            "human_box": human_box.tolist(),
            "prompt_box": variants[0].tolist(),
            "no_sam_output": True,
        })
        self._objects[object_id]["box"] = variants[0].copy()
        return variants[0]

    def _human_observation(
        self,
        frame_idx: int,
        object_id: int,
        human_box: np.ndarray,
        source: str,
    ) -> PromptObjectObservation:
        return PromptObjectObservation(
            frame_idx=frame_idx,
            sam_object_id=object_id,
            mask=np.zeros((1, 1), dtype=bool),
            box_xyxy=human_box.copy(),
            confidence=1.0,
            presence_score=1.0,
            source=source,
            is_human_verified=True,
        )

    def _add_human_to_cache(self, frame_idx: int, obs: PromptObjectObservation) -> None:
        cache = [
            o for o in self._output_cache.get(frame_idx, []) if o.sam_object_id != obs.sam_object_id
        ]
        cache.append(obs)
        self._output_cache[frame_idx] = cache

    def _parse_outputs(
        self, response: dict, frame_idx: int, source: str
    ) -> List[PromptObjectObservation]:
        raw = response.get("outputs", {})
        if isinstance(raw, list):
            return self._parse_output_items(raw, frame_idx, source)
        if not isinstance(raw, dict):
            return []
        obj_ids = self._as_numpy(raw.get("out_obj_ids"))
        boxes_xywh = self._as_numpy(raw.get("out_boxes_xywh"))
        masks = self._as_numpy(raw.get("out_binary_masks"))
        probs = self._as_numpy(raw.get("out_probs"))
        if obj_ids is None or boxes_xywh is None or masks is None:
            return []
        obj_ids = np.asarray(obj_ids).reshape(-1)
        boxes = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
        masks = np.asarray(masks)
        if masks.ndim == 3:
            masks = masks[:, None, :, :] if masks.shape[0] != 1 else masks[None]
        probs = (
            np.asarray(probs, dtype=float).reshape(-1)
            if probs is not None
            else np.full(len(obj_ids), 0.5, dtype=float)
        )
        observations: List[PromptObjectObservation] = []
        for i, oid in enumerate(obj_ids):
            if i >= len(boxes) or i >= len(masks):
                break
            nx, ny, nw, nh = boxes[i]
            x = nx * self._frame_w
            y = ny * self._frame_h
            w = nw * self._frame_w
            h = nh * self._frame_h
            box = np.asarray([x, y, x + w, y + h], dtype=float)
            mask = np.asarray(masks[i], dtype=bool)
            while mask.ndim > 2 and mask.shape[0] == 1:
                mask = mask[0]
            observations.append(
                PromptObjectObservation(
                    frame_idx=frame_idx,
                    sam_object_id=int(oid),
                    raw_sam_object_id=int(oid),
                    mask=mask,
                    box_xyxy=box,
                    confidence=float(probs[i]) if i < len(probs) else 0.5,
                    presence_score=float(probs[i]) if i < len(probs) else None,
                    source=source,
                    is_human_verified=source in ("human_add", "human_correction"),
                )
            )
        return observations

    def _parse_output_items(
        self, items: list, frame_idx: int, source: str
    ) -> List[PromptObjectObservation]:
        observations: List[PromptObjectObservation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            oid = item.get("obj_id", item.get("obj_ids"))
            if oid is None:
                continue
            mask = _extract_mask(item)
            box = _extract_box(item, mask)
            if mask is None or box is None:
                continue
            observations.append(
                PromptObjectObservation(
                    frame_idx=frame_idx,
                    sam_object_id=int(oid),
                    raw_sam_object_id=int(oid),
                    mask=mask,
                    box_xyxy=box,
                    confidence=float(item.get("score", item.get("scores", 0.5))),
                    presence_score=item.get("presence_score"),
                    source=source,
                    is_human_verified=source in ("human_add", "human_correction"),
                )
            )
        return observations

    @staticmethod
    def _as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)


def _extract_mask(item: dict) -> Optional[np.ndarray]:
    mask = item.get("mask", item.get("masks"))
    if mask is None:
        return None
    arr = np.asarray(mask)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr.astype(bool)


def _extract_box(item: dict, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    box = item.get("box_xyxy", item.get("box", item.get("boxes")))
    if box is not None:
        arr = np.asarray(box, dtype=float).reshape(-1)
        if arr.size == 4:
            return arr
    if mask is not None:
        return mask_to_box(mask)
    return None
