"""N6 frame-output assembler and validator."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameOutputRow:
    public_mot_id: int
    box_xyxy: np.ndarray
    confidence: float = 1.0


class FrameOutputValidationError(RuntimeError):
    pass


class FrameOutputAssembler:
    """Serialize exactly the authoritative final frame state.

    The assembler never appends corrected rows to pre rows.  It receives the
    transaction-committed active identity state and produces one row per
    public identity.
    """

    def assemble(
        self,
        frame_idx: int,
        rows: List[FrameOutputRow],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> List[FrameOutputRow]:
        violations = self.validate_frame_output(
            frame_idx, rows, frame_w=frame_w, frame_h=frame_h
        )
        if violations:
            raise FrameOutputValidationError(
                f"frame {frame_idx} output invalid: {violations}"
            )
        return list(rows)

    def validate_frame_output(
        self,
        frame_idx: int,
        rows: List[FrameOutputRow],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> List[str]:
        violations: List[str] = []
        seen_pid: Dict[int, int] = {}
        for i, row in enumerate(rows):
            if not isinstance(row.public_mot_id, int):
                violations.append(f"row {i}: non-int public id")
            if row.public_mot_id in seen_pid:
                violations.append(
                    f"row {i}: duplicate public id {row.public_mot_id} "
                    f"(also row {seen_pid[row.public_mot_id]})"
                )
            seen_pid[row.public_mot_id] = i
            box = np.asarray(row.box_xyxy, dtype=float)
            if box.size != 4:
                violations.append(f"row {i}: box size != 4")
                continue
            if not np.all(np.isfinite(box)):
                violations.append(f"row {i}: NaN/Inf box")
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                violations.append(f"row {i}: non-positive width/height")
            if frame_w is not None and (x1 < 0 or x2 > frame_w):
                violations.append(f"row {i}: box outside frame width")
            if frame_h is not None and (y1 < 0 or y2 > frame_h):
                violations.append(f"row {i}: box outside frame height")
        return violations

    def rows_to_mot(
        self,
        frame_idx: int,
        rows: List[FrameOutputRow],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> List[Tuple[int, int, float, float, float, float, float]]:
        rows = self.assemble(frame_idx, rows, frame_w=frame_w, frame_h=frame_h)
        out = []
        for row in sorted(rows, key=lambda r: r.public_mot_id):
            x1, y1, x2, y2 = np.asarray(row.box_xyxy, dtype=float)
            out.append(
                (
                    frame_idx + 1,
                    row.public_mot_id,
                    x1,
                    y1,
                    max(0.0, x2 - x1),
                    max(0.0, y2 - y1),
                    float(row.confidence),
                )
            )
        return out
