#!/usr/bin/env python
"""N8 temporal-error observer runner on the frozen P0 backbone (CPU)."""

import json
import os
import shutil
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import read_mot_rows
from sam3_intermot.interaction.n8_temporal_observer import N8Config, N8TemporalObserver


ROOT = Path(__file__).resolve().parents[1]
SEQ = os.environ["N8_SEQ"]
BUDGET = int(os.environ.get("N8_BUDGET", "0"))
OUT_DIR = Path(
    os.environ.get("N8_OUT_DIR", ROOT / "outputs/n8" / "tmp")
)
P0_SOURCE = Path(
    os.environ.get(
        "N8_P0_SOURCE",
        ROOT / "outputs/n5/integrity/canonical_mot_results/b0",
    )
)
DATASET_ROOT = Path(
    os.environ.get("N8_DATASET_ROOT", "/path/to/dancetrack")
)
FRAME_LIMIT = int(os.environ.get("N8_FRAMES", "0")) or None


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


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    dataset = DanceTrackDataset(
        str(DATASET_ROOT),
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
    cfg = N8Config(budget=BUDGET, sequence=SEQ)
    obs = N8TemporalObserver(backbone, gt_frames, num_frames, cfg, sequence=SEQ)
    obs.run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if BUDGET == 0:
        # B0 must be byte-identical to P0 (ZERO_INTERACTION_EQUIVALENCE).
        for sub in ("pre_mot", "post_mot"):
            dst = OUT_DIR / sub / f"{SEQ}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = P0_SOURCE / f"{SEQ}.txt"
            if FRAME_LIMIT is None:
                shutil.copyfile(src, dst)
            else:
                dst.write_text(
                    "".join(
                        line
                        for line in src.read_text(encoding="utf-8").splitlines(keepends=True)
                        if line.strip() and int(line.split(",")[0]) <= FRAME_LIMIT
                    ),
                    encoding="utf-8",
                )
    else:
        export_mot(OUT_DIR / "pre_mot" / f"{SEQ}.txt", obs.pre_rows)
        export_mot(OUT_DIR / "post_mot" / f"{SEQ}.txt", obs.post_rows)
    write_jsonl(OUT_DIR / "verified_errors.jsonl", obs.verified_errors)
    write_jsonl(OUT_DIR / "interaction_events.jsonl", obs.interaction_events)
    write_jsonl(OUT_DIR / "observer_memory_audit.jsonl", obs.observer_audit)
    write_jsonl(OUT_DIR / "system_state_hashes.jsonl", obs.state_hashes)
    by_type = {}
    for e in obs.verified_errors:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
    by_type_accepted = {}
    for e in obs.interaction_events:
        by_type_accepted[e["event_type"]] = by_type_accepted.get(e["event_type"], 0) + 1
    summary = {
        "sequence": SEQ,
        "budget": BUDGET,
        "num_frames": num_frames,
        "accepted_count": obs.accepted_count,
        "verified_errors_total": len(obs.verified_errors),
        "events_by_type": by_type,
        "accepted_by_type": by_type_accepted,
        "pre_rows": sum(len(v) for v in obs.pre_rows.values()),
        "post_rows": sum(len(v) for v in obs.post_rows.values()),
        "invariant_violations": obs.invariant_violations,
        "gt_audit": obs.gt_audit,
        "namespace_violations": obs.ns.violations(),
        "canonical_map_size": len(obs.canonical_map),
        "p0_sha256": __import__("hashlib").sha256(
            (P0_SOURCE / f"{SEQ}.txt").read_bytes()
        ).hexdigest(),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
