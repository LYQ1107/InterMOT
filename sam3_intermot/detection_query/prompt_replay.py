"""N13 PDR prompt-replay policies on the official SAM3.1 pipeline.

The official ``add_prompt`` API resets the session, so persistent prompting is
implemented by injecting per-frame geometric prompts into the inference state
before the official propagation loop processes each frame.  This is the
honest EXTERNAL_PERSISTENT_PROMPT_ROUTE: the prompts are model-generated
detector conditioning, not an internal latent detector query (see
docs/N13_SAM3_DETECTOR_QUERY_AUDIT.md).
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from sam3_intermot.adaptation.cfa_backend_runner import (
    CFABackendRunner,
    _iou,
    parse_raw_outputs,
)


def _to_cxcywh_norm(box_xyxy: np.ndarray, iw: int, ih: int):
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    cx = (x1 + x2) / 2.0 / iw
    cy = (y1 + y2) / 2.0 / ih
    w = (x2 - x1) / iw
    h = (y2 - y1) / ih
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(1e-4, w))
    h = min(1.0, max(1e-4, h))
    return np.asarray([cx, cy, w, h], dtype=float)


def set_frame_geometric_prompt(
    backend: CFABackendRunner, frame_idx: int, box_xyxy: Optional[np.ndarray]
) -> bool:
    """Inject (or clear) a per-frame person+box geometric prompt."""
    session = backend.backend._predictor._all_inference_states[
        backend.backend._session_id
    ]
    state = session["state"]
    model = backend.backend._predictor.model
    iw, ih = backend.backend._frame_w, backend.backend._frame_h
    if box_xyxy is None:
        state["per_frame_geometric_prompt"][frame_idx] = None
        state["per_frame_raw_box_input"][frame_idx] = None
        return False
    cxcywh = _to_cxcywh_norm(box_xyxy, iw, ih)
    boxes_cxcywh = torch.as_tensor(
        cxcywh.reshape(1, 4), dtype=torch.float32, device=state["device"]
    )
    box_labels = torch.as_tensor([1], dtype=torch.long, device=state["device"])
    _, _, geometric_prompt = model._get_visual_prompt(
        state, frame_idx, boxes_cxcywh, box_labels
    )
    state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
    return True


def invalidate_detector_prefetch(
    backend: CFABackendRunner, current_frame: int
) -> None:
    """Drop pre-fetched detector chunks for frames > current_frame.

    The official single-GPU detector pre-fetches the *next* frame's chunk with
    the *current* frame's geometric prompt (sam3_multiplex_detector.py Step 3
    in forward_video_grounding_multigpu).  Without invalidation, a prompt set
    for frame f+1 is silently ignored because the cached chunk was built with
    prompt(f).  We wait for pending async handles before removing entries.
    """
    state = backend.backend._predictor._all_inference_states[
        backend.backend._session_id
    ]["state"]
    mb = state["feature_cache"].get("multigpu_buffer")
    if not mb:
        return
    keys = [k for k in mb if k > current_frame]
    for k in keys:
        for _v, handle in mb[k].values():
            if handle is not None:
                handle.wait()
    for k in keys:
        del mb[k]


def expand_box(box: np.ndarray, scale: float, iw: int, ih: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return np.asarray(
        [
            max(0.0, cx - w / 2),
            max(0.0, cy - h / 2),
            min(iw, cx + w / 2),
            min(ih, cy + h / 2),
        ],
        dtype=float,
    )


@dataclass
class FrameRecord:
    frame: int
    cand_boxes: List[np.ndarray] = field(default_factory=list)
    prompt_box: Optional[np.ndarray] = None
    delivered_box: Optional[np.ndarray] = None
    lifecycle: str = ""


@dataclass
class PDREpisode:
    sequence: str
    frame: int
    event_type: str
    gid: int
    human_box: np.ndarray
    policy: str
    prompt_had_output: bool = False
    records: Dict[int, FrameRecord] = field(default_factory=dict)
    seconds: float = 0.0


class PromptPolicy:
    """Per-frame policy interface: given history up to frame f, return the
    prompt box for frame f (None = no detector box prompt)."""

    def __init__(self, iw: int, ih: int):
        self.iw, self.ih = iw, ih
        self.last_trusted: Optional[np.ndarray] = None
        self.prev_trusted: Optional[np.ndarray] = None
        self.last_trusted_frame: Optional[int] = None
        self.miss_streak = 0

    def seed(self, box: np.ndarray, frame_idx: int):
        self.last_trusted = np.asarray(box, dtype=float).copy()
        self.last_trusted_frame = frame_idx

    def observe(self, frame_idx: int, cands: List[np.ndarray], delivered: Optional[np.ndarray]):
        self.miss_streak = 0 if delivered is not None else self.miss_streak + 1
        if delivered is not None:
            self.prev_trusted = self.last_trusted
            self.last_trusted = delivered.copy()
            self.last_trusted_frame = frame_idx

    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        raise NotImplementedError


class OneShotPolicy(PromptPolicy):
    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        return None


class LastBoxPolicy(PromptPolicy):
    """A1: repeat the last trusted box (model-generated)."""

    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        return None if self.last_trusted is None else self.last_trusted.copy()


class MotionPolicy(PromptPolicy):
    """A2: linear velocity extrapolation from the last two trusted boxes."""

    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.last_trusted is None:
            return None
        if self.prev_trusted is None or self.last_trusted_frame is None:
            return self.last_trusted.copy()
        dx = self.last_trusted - self.prev_trusted
        pred = self.last_trusted + dx * max(1, frame_idx - self.last_trusted_frame)
        pred = np.clip(
            pred,
            [0, 0, 0, 0],
            [self.iw, self.ih, self.iw, self.ih],
        )
        return pred


class GatedPolicy(PromptPolicy):
    """A3: update the trusted box only on confident IoU continuity; on a miss
    hold the last trusted box and expand the search prompt (DORMANT)."""

    def __init__(self, iw: int, ih: int, update_iou: float = 0.5,
                 expand_after: int = 3, expand_scale: float = 1.5):
        super().__init__(iw, ih)
        self.update_iou = update_iou
        self.expand_after = expand_after
        self.expand_scale = expand_scale

    def observe(self, frame_idx, cands, delivered):
        self.miss_streak = 0 if delivered is not None else self.miss_streak + 1
        if delivered is not None and self.last_trusted is not None:
            if _iou(delivered, self.last_trusted) >= self.update_iou:
                self.prev_trusted = self.last_trusted
                self.last_trusted = delivered.copy()
                self.last_trusted_frame = frame_idx

    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.last_trusted is None:
            return None
        if self.miss_streak >= self.expand_after:
            return expand_box(self.last_trusted, self.expand_scale, self.iw, self.ih)
        return self.last_trusted.copy()


class OraclePolicy(PromptPolicy):
    """Oracle future-GT diagnostic only (never deployed)."""

    def __init__(self, iw: int, ih: int, gt: Dict[int, object], gid: int):
        super().__init__(iw, ih)
        self.gt = gt
        self.gid = gid

    def prompt_for(self, frame_idx: int) -> Optional[np.ndarray]:
        entry = self.gt.get(frame_idx)
        if entry is None or self.gid not in entry.gt_ids:
            return None
        return np.asarray(entry.boxes[entry.gt_ids.index(self.gid)], dtype=float)


POLICY_FACTORIES: Dict[str, Callable] = {
    "one_shot": OneShotPolicy,
    "last_box": LastBoxPolicy,
    "motion": MotionPolicy,
    "gated": GatedPolicy,
    "oracle": OraclePolicy,
}


def _best_delivery(
    prev: Optional[np.ndarray], cands: List[np.ndarray]
) -> Optional[np.ndarray]:
    if not cands:
        return None
    if prev is None:
        return cands[0].copy()
    best = max(cands, key=lambda c: _iou(c, prev))
    if _iou(best, prev) >= 0.3:
        return best.copy()
    return None


def run_pdr_episode(
    runner: CFABackendRunner,
    sequence: str,
    frame_idx: int,
    event_type: str,
    gid: int,
    human_box: np.ndarray,
    policy: str,
    gt: Dict[int, object],
    horizon: int = 30,
) -> PDREpisode:
    """Run one human-seeded episode with the given prompt policy.

    One human interaction at ``frame_idx`` (person+box); then the policy
    generates detector prompts for frames t+1..t+H.  GT is used only for
    evaluation and for the ``oracle`` diagnostic policy.
    """
    backend = runner._ensure_backend()
    video = str(
        Path("/path/to/dancetrack")
        / runner.split
        / sequence
        / "img1"
    )
    backend.start_video(video)
    # The official builder enables 16-frame batched grounding, which pre-computes
    # detector chunks with a single geometric prompt and therefore cannot apply
    # per-frame prompts.  Disabling it is an adapter-level runtime control (no
    # third-party source modification) that makes the per-frame detector prompt
    # route causal.
    if getattr(backend._predictor.model, "use_batched_grounding", False):
        backend._predictor.model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h
    ep = PDREpisode(
        sequence=sequence,
        frame=frame_idx,
        event_type=event_type,
        gid=gid,
        human_box=np.asarray(human_box, dtype=float),
        policy=policy,
    )
    t0 = time.time()

    pol = POLICY_FACTORIES[policy](iw, ih) if policy != "oracle" else OraclePolicy(iw, ih, gt, gid)
    pol.seed(ep.human_box, frame_idx)

    box = ep.human_box
    x1, y1, x2, y2 = box
    req_prompt = dict(
        type="add_prompt",
        session_id=backend._session_id,
        frame_index=frame_idx,
        text="person",
        bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
        bounding_box_labels=[1],
        clear_old_boxes=True,
    )
    prompt_resp = backend._predictor.handle_request(req_prompt)
    cands0 = parse_raw_outputs(prompt_resp, frame_size=(iw, ih))
    ep.prompt_had_output = len(cands0) > 0
    # The official add_prompt->propagate flow records an "add" action and then
    # chooses SAM2 partial propagation, which never admits future detector
    # boxes (the N12-observed causal disconnect).  For N13's detector-side
    # question we must run full-VG propagation, which re-runs the detector per
    # frame and admits new detections.  Clearing the runtime action history is
    # an adapter-level control (no third-party source modification).
    session_state = backend._predictor._all_inference_states[
        backend._session_id
    ]["state"]
    session_state["action_history"].clear()

    prev = None
    if cands0:
        best = max(cands0, key=lambda c: _iou(c[1], box))
        if _iou(best[1], box) >= 0.3:
            prev = best[1].copy()
    if prev is None:
        prev = box.copy()
    pol.observe(frame_idx, [b for _, b in cands0], prev)
    ep.records[frame_idx] = FrameRecord(
        frame=frame_idx,
        cand_boxes=[b for _, b in cands0],
        prompt_box=box.copy(),
        delivered_box=prev.copy(),
    )

    # Map frame -> prompt box actually set for that frame (None = no prompt).
    prompt_map: Dict[int, Optional[np.ndarray]] = {frame_idx: box.copy()}
    next_frame = frame_idx + 1
    prompt_map[next_frame] = pol.prompt_for(next_frame)
    set_frame_geometric_prompt(runner, next_frame, prompt_map[next_frame])

    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=frame_idx,
        # Full-video tracking bound (same proven path as N12).  The caller
        # stops the lazy generator after t+H, so only the needed frames are
        # actually processed.  A short max_frame_num_to_track triggers a
        # SAM3 tracker memory-conditioning bug at this pinned commit.
        max_frame_num_to_track=None,
    )
    try:
        for response in backend._predictor.handle_stream_request(request=req):
            f = int(response["frame_index"])
            cands = parse_raw_outputs(response, frame_size=(iw, ih))
            cand_boxes = [b for _, b in cands]
            delivered = _best_delivery(prev, cand_boxes)
            pol.observe(f, cand_boxes, delivered)
            if delivered is not None:
                prev = delivered.copy()
            ep.records[f] = FrameRecord(
                frame=f,
                cand_boxes=cand_boxes,
                prompt_box=prompt_map.get(f),
                delivered_box=delivered,
            )
            if f >= frame_idx + horizon:
                break
            nf = f + 1
            prompt_map[nf] = pol.prompt_for(nf)
            set_frame_geometric_prompt(runner, nf, prompt_map[nf])
            invalidate_detector_prefetch(runner, f)
    finally:
        try:
            backend.close()
        except Exception:
            pass
    ep.seconds = time.time() - t0
    return ep


def admission_hit(ep: PDREpisode, gt: Dict[int, object], f: int) -> bool:
    entry = gt.get(f)
    if entry is None or ep.gid not in entry.gt_ids:
        return False
    gb = np.asarray(entry.boxes[entry.gt_ids.index(ep.gid)], dtype=float)
    return any(_iou(c, gb) >= 0.5 for c in ep.records.get(f, FrameRecord(f)).cand_boxes)


def delivered_hit(ep: PDREpisode, gt: Dict[int, object], f: int) -> bool:
    entry = gt.get(f)
    rec = ep.records.get(f)
    if entry is None or rec is None or rec.delivered_box is None:
        return False
    if ep.gid not in entry.gt_ids:
        return False
    gb = np.asarray(entry.boxes[entry.gt_ids.index(ep.gid)], dtype=float)
    return _iou(rec.delivered_box, gb) >= 0.5


def false_capture(ep: PDREpisode, gt: Dict[int, object], f: int) -> bool:
    """Delivered box belongs to a different GT identity."""
    rec = ep.records.get(f)
    if rec is None or rec.delivered_box is None:
        return False
    entry = gt.get(f)
    if entry is None:
        return False
    db = rec.delivered_box
    target_iou = 0.0
    if ep.gid in entry.gt_ids:
        gb = np.asarray(entry.boxes[entry.gt_ids.index(ep.gid)], dtype=float)
        target_iou = _iou(db, gb)
    for other, ob in zip(entry.gt_ids, entry.boxes):
        if other == ep.gid:
            continue
        ob = np.asarray(ob, dtype=float)
        if _iou(db, ob) >= 0.5 and _iou(db, ob) > target_iou:
            return True
    return False


def recall_at(ep: PDREpisode, gt: Dict[int, object], h: int,
              metric: str = "delivered") -> float:
    fs = [f for f in range(ep.frame + 1, ep.frame + h + 1)]
    hits = n = 0
    for f in fs:
        entry = gt.get(f)
        if entry is None or ep.gid not in entry.gt_ids:
            continue
        n += 1
        if metric == "admission":
            hits += int(admission_hit(ep, gt, f))
        else:
            hits += int(delivered_hit(ep, gt, f))
    return hits / max(1, n)
