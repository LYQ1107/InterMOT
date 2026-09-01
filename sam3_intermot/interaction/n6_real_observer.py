"""N6 real-SAM observer with official segment-restart propagation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.interaction.n6_observer import N6BackboneObserver, N6Config
from sam3_intermot.interaction.simulator import GTFrame


class N6RealObserver(N6BackboneObserver):
    """Continuous observer over the real SAM 3.1 backend.

    Segment protocol:
    - At every segment start the official ``reset_session`` is called, all
      active user identities are re-prompted with their authoritative boxes
      (fresh ``sam_object_id``), then ``detect_concept`` + ``propagate`` fill
      the segment.
    - Identity transactions (REASSIGN/SWAP/RECOVER/CORRECT display) apply
      immediately through the namespace.
    - SAM-affecting changes (new objects, corrected prompt boxes) take effect
      at the next segment re-prompt (<= segment_len frames).
    - Public MOT ids never change across segments.
    """

    def __init__(
        self,
        backend,
        video_source: str,
        gt_frames: Dict[int, GTFrame],
        num_frames: int,
        config: N6Config,
        sequence: str = "",
        segment_len: int = 30,
    ) -> None:
        super().__init__({}, gt_frames, num_frames, config, sequence=sequence)
        self.backend = backend
        self.video_source = video_source
        self.segment_len = segment_len
        self.propagated: Dict[int, list] = {}
        self.active_user_box: Dict[int, np.ndarray] = {}
        self.segment_id = 0
        self.segment_restarts = 0
        self._next_sam_ext = 20000

    def run(self) -> None:
        self.backend.start_video(self.video_source)
        try:
            self._ensure_segment(0)
            for f in range(self.num_frames):
                if f not in self.propagated:
                    self._ensure_segment(f)
                self._process_frame(f)
                # after processing, remember authoritative boxes for restart
                for pid, box in self.post_rows.get(f, []):
                    uid = self.ns.user_for_public(pid)
                    if uid is not None:
                        self.active_user_box[uid] = np.asarray(box, dtype=float).copy()
                if (f + 1) % self.segment_len == 0 and f + 1 < self.num_frames:
                    self._ensure_segment(f + 1)
        finally:
            self.backend.close()

    def _ensure_segment(self, start: int) -> None:
        self.segment_id += 1
        self.backend.reset_session()
        self.propagated = {}
        for uid, box in sorted(self.active_user_box.items()):
            sam_id = self._next_sam_ext
            self._next_sam_ext += 1
            self.backend.add_box(start, sam_id, box)
            self.ns.bind_sam(self.segment_id, sam_id, uid)
            self.ns.bind_auto(sam_id, uid)
        dets = self.backend.detect_concept(start, "person")
        self.propagated[start] = list(dets)
        end = min(start + self.segment_len - 1, self.num_frames - 1)
        prop = self.backend.propagate(
            start, end, start_frame_index=start, keep_masks=False
        )
        for f, obs_list in prop.items():
            self.propagated[f] = list(obs_list)
        self.backend._output_cache.clear()
        self.segment_restarts += 1

    def _raw(self, frame_idx: int) -> List[Tuple[int, np.ndarray]]:
        out = []
        for obs in self.propagated.get(frame_idx, []):
            sam = obs.sam_object_id
            out.append((sam, np.asarray(obs.box_xyxy, dtype=float).copy()))
        return out
