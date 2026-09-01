"""Build N9 tracklet-relinking episodes for a sequence set (CPU)."""

import csv
import json
import os
from pathlib import Path

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import read_mot_rows
from sam3_intermot.n9.relink_benchmark import (
    assign_gt,
    build_episodes,
    build_decision_episodes,
    build_segments,
    write_episodes_csv,
    write_decision_csv,
)


ROOT = Path(".")
SPLIT = os.environ.get("N9_SPLIT", "train")
SEQS = sorted(os.environ.get("N9_SEQS", "").split()) or sorted(
    p.name
    for p in (Path("/path/to/dancetrack") / SPLIT).iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
P0_DIR = Path(
    os.environ.get(
        "N9_P0_DIR",
        str(ROOT / "outputs/n9/p0_train")
        if SPLIT == "train"
        else str(ROOT / "outputs/n5/integrity/canonical_mot_results/b0"),
    )
)
OUT = Path(os.environ.get("N9_BENCH_DIR", ROOT / "outputs/n9/benchmark" / SPLIT))


def main() -> None:
    ds = DanceTrackDataset("/path/to/dancetrack", split=SPLIT)
    total = 0
    rows_out = []
    for seq in SEQS:
        p0_path = P0_DIR / f"{seq}.txt"
        if not p0_path.exists():
            print(json.dumps({"sequence": seq, "status": "SKIP_NO_P0"}))
            continue
        p0 = read_mot_rows(p0_path)
        gt = ds.load_gt(seq)
        segs = build_segments(p0)
        assign_gt(segs, gt)
        episodes = build_decision_episodes(seq, p0, gt)
        out_csv = OUT / f"{seq}.csv"
        write_decision_csv(out_csv, episodes)
        total += len(episodes)
        print(
            json.dumps(
                {
                    "sequence": seq,
                    "segments": len(segs),
                    "episodes": len(episodes),
                    "miss_episodes": sum(1 for e in episodes if e["miss"]),
                    "tid_changed": sum(1 for e in episodes if e["tid_changed"]),
                }
            )
        )
        rows_out.append(
            {
                "sequence": seq,
                "episodes": len(episodes),
                "segments": len(segs),
            }
        )
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "episodes", "segments"])
        w.writeheader()
        w.writerows(rows_out)
    (OUT / "total.json").write_text(
        json.dumps({"split": SPLIT, "sequences": len(SEQS), "episodes": total}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"split": SPLIT, "sequences": len(SEQS), "episodes": total}))


if __name__ == "__main__":
    main()
