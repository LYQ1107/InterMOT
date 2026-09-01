#!/usr/bin/env python
"""N6 CPU backbone observer runner (P0 frozen backbone + identity namespace)."""

import json
import os
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import read_mot_rows
from sam3_intermot.interaction.n6_observer import N6BackboneObserver, N6Config


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = os.environ["N6_PROTOCOL"]
BUDGET = int(os.environ.get("N6_BUDGET", "0"))
SEQ = os.environ["N6_SEQ"]
OUT_DIR = Path(os.environ.get("N6_OUT_DIR", ROOT / "outputs/n6" / "tmp"))
P0_SOURCE = Path(
    os.environ.get(
        "N6_P0_SOURCE",
        ROOT / "outputs/n5/integrity/canonical_mot_results/b0",
    )
)
FRAME_LIMIT = int(os.environ.get("N6_FRAMES", "0")) or None


def export_mot(path: Path, rows_by_frame: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(rows_by_frame):
        for pid, box in sorted(rows_by_frame[f], key=lambda kv: kv[0]):
            x1, y1, x2, y2 = np.asarray(box, dtype=float)
            lines.append(
                f"{f+1},{pid},{x1:.2f},{y1:.2f},{max(0.0, x2-x1):.2f},"
                f"{max(0.0, y2-y1):.2f},1.000,-1,-1,-1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    dataset = DanceTrackDataset(
        str(Path("/path/to/dancetrack")),
        sequences=[SEQ],
        split="val",
    )
    num_frames = dataset.num_frames(SEQ)
    gt_frames = dataset.load_gt(SEQ)
    backbone: dict = {}
    all_rows = read_mot_rows(P0_SOURCE / f"{SEQ}.txt")
    for f in range(num_frames):
        backbone[f] = list(all_rows.get(f, []))
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)
        gt_frames = {f: v for f, v in gt_frames.items() if f < num_frames}
        backbone = {f: v for f, v in backbone.items() if f < num_frames}
    cfg = N6Config(
        protocol=PROTOCOL,
        budget=BUDGET,
        correct_localization=PROTOCOL == "p2",
        correct_false_track=PROTOCOL == "p2",
        stateful=True,
    )
    obs = N6BackboneObserver(backbone, gt_frames, num_frames, cfg, sequence=SEQ)
    obs.run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_mot(OUT_DIR / "pre_mot" / f"{SEQ}.txt", obs.pre_rows)
    export_mot(OUT_DIR / "post_mot" / f"{SEQ}.txt", obs.post_rows)
    with (OUT_DIR / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in obs.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    by_type = {}
    for e in obs.events:
        by_type[e["action_type"]] = by_type.get(e["action_type"], 0) + 1
    summary = {
        "sequence": SEQ,
        "protocol": PROTOCOL,
        "budget": BUDGET,
        "num_frames": num_frames,
        "total_commands": len(obs.events),
        "accepted_commands": obs.accepted_count,
        "by_type": by_type,
        "pre_rows": sum(len(v) for v in obs.pre_rows.values()),
        "post_rows": sum(len(v) for v in obs.post_rows.values()),
        "invariant_violations": obs.invariant_violations,
        "allocations_total": obs.ns.allocator.allocations_total,
        "allocations_by_action": obs.ns.allocator.allocations_by_action,
        "namespace_violations": obs.ns.violations(),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "state_hashes.json").write_text(
        json.dumps(obs.state_hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
