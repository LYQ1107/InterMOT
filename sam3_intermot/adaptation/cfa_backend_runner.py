"""CFA no-update baseline using the official full SAM3.1 pipeline.

The project's Sam3Backend parser drops some add_prompt responses (outputs are a
dict, not a list).  This runner uses the same predictor but parses raw outputs
permissively, then greedily tracks the corrected object by box overlap.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.observations.mask_to_box import mask_to_box


DT = Path("/path/to/dancetrack")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_item(item: dict) -> Optional[Tuple[int, np.ndarray]]:
    if not isinstance(item, dict):
        return None
    oid = item.get("obj_id", item.get("obj_ids"))
    if oid is None:
        return None
    mask = item.get("mask", item.get("masks"))
    box = item.get("box_xyxy", item.get("box", item.get("boxes")))
    if box is not None:
        arr = np.asarray(box, dtype=float).reshape(-1)
        if arr.size == 4:
            return int(oid), arr
    if mask is not None:
        arr = np.asarray(mask)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            b = mask_to_box(arr > 0)
            if b is not None:
                return int(oid), b
    return None


def parse_raw_outputs(
    response: dict, frame_size: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, np.ndarray]]:
    raw = response.get("outputs", {})
    out = []
    if isinstance(raw, dict):
        if "out_obj_ids" in raw and "out_boxes_xywh" in raw:
            oids = np.asarray(raw["out_obj_ids"]).reshape(-1)
            boxes = np.asarray(raw["out_boxes_xywh"], dtype=float).reshape(-1, 4)
            iw = ih = 1.0
            if frame_size is not None:
                iw, ih = frame_size
            for i, oid in enumerate(oids):
                if i >= len(boxes):
                    break
                nx, ny, nw, nh = boxes[i]
                out.append(
                    (
                        int(oid),
                        np.asarray(
                            [nx * iw, ny * ih, (nx + nw) * iw, (ny + nh) * ih],
                            dtype=float,
                        ),
                    )
                )
            return out
        for key, item in raw.items():
            parsed = _extract_item(item) if isinstance(item, dict) else None
            if parsed is None:
                # maybe item is a nested list of dicts
                if isinstance(item, (list, tuple)):
                    for it in item:
                        p = _extract_item(it)
                        if p is not None:
                            out.append(p)
                continue
            out.append(parsed)
    elif isinstance(raw, list):
        for item in raw:
            p = _extract_item(item)
            if p is not None:
                out.append(p)
    return out


@dataclass
class CFAEpisode:
    sequence: str
    frame: int
    event_type: str
    gid: int
    human_box: List[float]
    prompt_had_output: bool = False
    frames: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)


class CFABackendRunner:
    def __init__(self, checkpoint_path: str, split: str = "train", **backend_kwargs):
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.backend_kwargs = backend_kwargs
        self.backend: Optional[Sam3Backend] = None
        self.dataset = DanceTrackDataset(str(DT), sequences=[], split=split)

    def _ensure_backend(self) -> Sam3Backend:
        if self.backend is None:
            self.backend = Sam3Backend(
                checkpoint_path=self.checkpoint_path,
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=True,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
                **self.backend_kwargs,
            )
        return self.backend

    def run_episode(
        self, sequence: str, frame_idx: int, event_type: str, gid: int,
        human_box: np.ndarray, horizon: int = 30,
    ) -> CFAEpisode:
        backend = self._ensure_backend()
        video = str(DT / self.split / sequence / "img1")
        backend.start_video(video)
        ep = CFAEpisode(
            sequence=sequence,
            frame=frame_idx,
            event_type=event_type,
            gid=gid,
            human_box=list(np.asarray(human_box, dtype=float)),
        )
        # Prompt with the human box directly (bypass the strict backend parser).
        box = np.asarray(human_box, dtype=float)
        x1, y1, x2, y2 = box
        iw, ih = backend._frame_w, backend._frame_h
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
        if not ep.prompt_had_output and os.environ.get("N12_DEBUG_PROMPT"):
            raw = prompt_resp.get("outputs")
            print(
                "N12_DEBUG_PROMPT type/len/keys:",
                type(raw).__name__,
                len(raw) if raw is not None else None,
                list(raw.keys())[:6] if isinstance(raw, dict) else None,
                flush=True,
            )
            if isinstance(raw, dict):
                for k in ("out_obj_ids", "out_boxes_xywh", "out_binary_masks"):
                    v = raw.get(k)
                    print(
                        k,
                        type(v).__name__,
                        getattr(v, "shape", None),
                        repr(v)[:120] if not hasattr(v, "shape") else "",
                        flush=True,
                    )
        prev = None
        if cands0:
            best = max(cands0, key=lambda c: _iou(c[1], box))
            prev = best[1] if _iou(best[1], box) >= 0.3 else None
        req = dict(
            type="propagate_in_video",
            session_id=backend._session_id,
            propagation_direction="forward",
            start_frame_index=frame_idx,
        )
        try:
            for response in backend._predictor.handle_stream_request(request=req):
                f = int(response["frame_index"])
                if f > frame_idx + horizon:
                    break
                if f < frame_idx:
                    continue
                cands = parse_raw_outputs(response, frame_size=(iw, ih))
                if cands:
                    if prev is not None:
                        best = max(cands, key=lambda c: _iou(c[1], prev))
                        prev = best[1] if _iou(best[1], prev) >= 0.3 else None
                else:
                    prev = None
                ep.frames[f] = prev
        finally:
            try:
                backend.close()
            except Exception:
                pass
        return ep

    def close(self):
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None


def recall_at(ep: CFAEpisode, gt: Dict[int, object], h: int) -> float:
    fs = [f for f in range(ep.frame + 1, ep.frame + h + 1)]
    hits = 0
    n = 0
    for f in fs:
        entry = gt.get(f)
        if entry is None or ep.gid not in entry.gt_ids:
            continue
        n += 1
        b = ep.frames.get(f)
        if b is not None:
            gb = entry.boxes[entry.gt_ids.index(ep.gid)]
            if _iou(b, gb) >= 0.5:
                hits += 1
    return hits / max(1, n)
