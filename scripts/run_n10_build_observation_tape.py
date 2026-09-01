#!/usr/bin/env python
"""Build N10 anonymous ObservationTapes for train + val sequences."""

import json
import os
import time
from pathlib import Path

import numpy as np

from sam3_intermot.association.observation_tape import build_tape, save_tape
from sam3_intermot.datasets.dancetrack import DanceTrackDataset


ROOT = Path(".")
DT = Path("/path/to/dancetrack")
SPLITS = os.environ.get("N10_SPLITS", "train val").split()
OUT = ROOT / "outputs/n10/tapes"
P0_TRAIN = ROOT / "outputs/n9/p0_train"
P0_VAL = ROOT / "outputs/n5/integrity/canonical_mot_results/b0"
FEAT = ROOT / "outputs/n9/features"


def main() -> None:
    manifest = []
    for split in SPLITS:
        ds = DanceTrackDataset(str(DT), split=split)
        p0_dir = P0_TRAIN if split == "train" else P0_VAL
        seqs = sorted(
            p.name
            for p in (DT / split).iterdir()
            if p.is_dir() and (p / "gt" / "gt.txt").is_file()
        )
        for seq in seqs:
            num_frames = ds.num_frames(seq)
            t0 = time.time()
            tape = build_tape(seq, p0_dir / f"{seq}.txt", FEAT / f"{seq}.npz", num_frames)
            save_tape(OUT / f"{seq}.npz", tape)
            manifest.append(
                {
                    "sequence": seq,
                    "split": split,
                    "num_frames": num_frames,
                    "rows": int(len(tape["frame"])),
                    "feat_ratio": round(float(np.mean(tape["has_feat"])), 4),
                    "wall_seconds": round(time.time() - t0, 2),
                }
            )
            print(json.dumps(manifest[-1]), flush=True)
    (OUT / "observation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tapes": len(manifest), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
