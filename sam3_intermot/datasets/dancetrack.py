"""DanceTrack reading protocol reused read-only from the existing project."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sam3_intermot.interaction.simulator import GTFrame


class DanceTrackDataset:
    """Read-only accessor for DanceTrack sequences and GT."""

    def __init__(
        self,
        root: str,
        sequences: Optional[List[str]] = None,
        seqmap_path: Optional[str] = None,
        split: str = "val",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.seqmap_path = Path(seqmap_path) if seqmap_path else None
        if sequences is None:
            sequences = self._read_seqmap_or_default()
        self.sequences = sequences

    def _read_seqmap_or_default(self) -> List[str]:
        if self.seqmap_path is not None and self.seqmap_path.exists():
            lines = self.seqmap_path.read_text(encoding="utf-8").splitlines()
            seqs = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split()[0]
                seqs.append(name)
            if seqs:
                return seqs
        split_dir = self.root / self.split
        if split_dir.is_dir():
            return sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        return []

    def sequence_dir(self, sequence: str) -> Path:
        return self.root / self.split / sequence

    def frame_paths(self, sequence: str) -> List[Path]:
        seq_dir = self.sequence_dir(sequence)
        if not seq_dir.is_dir():
            return []
        return sorted(seq_dir.glob("img1/*.jpg"))

    def num_frames(self, sequence: str) -> int:
        return len(self.frame_paths(sequence))

    def load_gt(self, sequence: str) -> Dict[int, GTFrame]:
        """Load MOT-format GT into frame-limited GTFrame objects."""
        gt_path = self.sequence_dir(sequence) / "gt" / "gt.txt"
        frames: Dict[int, GTFrame] = {}
        if not gt_path.exists():
            return frames
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_1based = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            if w <= 0 or h <= 0:
                continue
            frame_idx = frame_1based - 1
            entry = frames.setdefault(frame_idx, GTFrame())
            entry.gt_ids.append(track_id)
            entry.boxes.append(np.asarray([x, y, x + w, y + h], dtype=float))
        return frames


def find_frozen_sequence_list(existing_intermot_root: str) -> Optional[str]:
    """Locate the frozen DanceTrack sequence list in InterMOT (read-only)."""
    root = Path(existing_intermot_root)
    candidates = [
        root / "configs" / "dancetrack_validate.txt",
        root / "configs" / "dancetrack_seqmap.txt",
        root / "datasets" / "dancetrack_validate.txt",
        root / "outputs" / "dancetrack_validate.txt",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None
