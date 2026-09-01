"""N10 anonymous observation tape: vision output decoupled from identity."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from sam3_intermot.interaction.continuous_observer import read_mot_rows


def build_tape(
    sequence: str,
    p0_path: Path,
    feat_npz: Path,
    num_frames: int,
) -> Dict[str, np.ndarray]:
    """Build one ObservationTape from frozen P0 rows + cached ReID features.

    Every P0 row is kept (boxes are the frozen visual output).  Native tid is
    retained only as an optional cue; it never becomes the public identity.
    Rows without a cached ReID feature get a zero feature and has_feat=0.
    """
    rows = read_mot_rows(p0_path)
    first_seen: Dict[int, int] = {}
    frames: List[int] = []
    obs_ids: List[int] = []
    boxes: List[np.ndarray] = []
    confs: List[float] = []
    feats: List[np.ndarray] = []
    has_feat: List[float] = []
    native_tids: List[int] = []
    native_ages: List[float] = []
    d = np.load(feat_npz)
    feat_map = {
        (int(f), int(t)): i
        for i, (f, t) in enumerate(zip(d["frame"], d["tid"]))
    }
    feat_cache = d["feat"].astype(np.float32)
    for f in range(num_frames):
        fr = rows.get(f, [])
        for local, (tid, box) in enumerate(sorted(fr, key=lambda kv: kv[0])):
            if int(tid) not in first_seen:
                first_seen[int(tid)] = f
            frames.append(f)
            obs_ids.append(local)
            boxes.append(np.asarray(box, dtype=np.float32))
            confs.append(1.0)
            fi = feat_map.get((f, int(tid)))
            if fi is None:
                feats.append(np.zeros(512, dtype=np.float32))
                has_feat.append(0.0)
            else:
                v = feat_cache[fi]
                n = float(np.linalg.norm(v))
                feats.append(v / n if n > 1e-6 else np.zeros(512, dtype=np.float32))
                has_feat.append(1.0)
            native_tids.append(int(tid))
            native_ages.append(float(f - first_seen[int(tid)]))
    return {
        "frame": np.asarray(frames, dtype=np.int32),
        "obs_id": np.asarray(obs_ids, dtype=np.int32),
        "box": np.asarray(boxes, dtype=np.float32),
        "conf": np.asarray(confs, dtype=np.float32),
        "feat": np.asarray(feats, dtype=np.float32),
        "has_feat": np.asarray(has_feat, dtype=np.float32),
        "native_tid": np.asarray(native_tids, dtype=np.int32),
        "native_age": np.asarray(native_ages, dtype=np.float32),
    }


def save_tape(path: Path, tape: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tape)


def load_tape(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def tape_rows_by_frame(tape: Dict[str, np.ndarray]) -> Dict[int, List[dict]]:
    """Group tape arrays into per-frame lists of lightweight obs dicts."""
    out: Dict[int, List[dict]] = {}
    for i in range(len(tape["frame"])):
        f = int(tape["frame"][i])
        out.setdefault(f, []).append(
            {
                "obs_id": int(tape["obs_id"][i]),
                "box": np.asarray(tape["box"][i], dtype=np.float32),
                "conf": float(tape["conf"][i]),
                "feat": np.asarray(tape["feat"][i], dtype=np.float32),
                "has_feat": float(tape["has_feat"][i]),
                "native_tid": int(tape["native_tid"][i]),
                "native_age": float(tape["native_age"][i]),
                "human_obs": 0.0,
            }
        )
    return out
