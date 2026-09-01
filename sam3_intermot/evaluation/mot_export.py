"""Standard MOTChallenge export and validation."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam3_intermot.backend.output_types import PromptObjectObservation


def observation_to_mot_row(
    frame_1based: int, track_id: int, obs: PromptObjectObservation
) -> Tuple[int, int, float, float, float, float, float]:
    x1, y1, x2, y2 = obs.box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (frame_1based, track_id, x1, y1, w, h, float(obs.confidence))


def export_mot_file(
    path: Path | str,
    outputs_by_frame: Dict[int, List[Tuple[int, PromptObjectObservation]]],
    start_frame: int = 0,
) -> None:
    """Write ``frame, id, x, y, w, h, score, -1, -1, -1`` rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame_idx in sorted(outputs_by_frame):
        frame_1based = frame_idx + 1
        for track_id, obs in sorted(outputs_by_frame[frame_idx], key=lambda kv: kv[0]):
            r = observation_to_mot_row(frame_1based, track_id, obs)
            lines.append(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.3f},-1,-1,-1")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_mot_file(
    path: Path | str,
    num_frames: Optional[int] = None,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> List[str]:
    violations: List[str] = []
    p = Path(path)
    if not p.exists():
        return ["file_missing"]
    seen = set()
    max_frame = 0
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 7:
            violations.append(f"line {line_no}: wrong column count")
            continue
        try:
            frame, tid, x, y, w, h, score = (float(parts[0]), int(float(parts[1])),
                                             float(parts[2]), float(parts[3]),
                                             float(parts[4]), float(parts[5]), float(parts[6]))
        except ValueError:
            violations.append(f"line {line_no}: non-numeric fields")
            continue
        if not all(np.isfinite(v) for v in (x, y, w, h, score)):
            violations.append(f"line {line_no}: NaN/Inf")
        if w <= 0 or h <= 0:
            violations.append(f"line {line_no}: non-positive w/h")
        if frame_w is not None and (x < 0 or x + w > frame_w):
            violations.append(f"line {line_no}: box outside frame width")
        if frame_h is not None and (y < 0 or y + h > frame_h):
            violations.append(f"line {line_no}: box outside frame height")
        key = (int(frame), tid)
        if key in seen:
            violations.append(f"line {line_no}: duplicate (frame, id)")
        seen.add(key)
        max_frame = max(max_frame, int(frame))
    if num_frames is not None and max_frame > num_frames:
        violations.append(f"max frame {max_frame} exceeds sequence length {num_frames}")
    return violations
