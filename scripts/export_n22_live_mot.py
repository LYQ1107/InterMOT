#!/usr/bin/env python
"""Export N22 true-live event traces to MOTChallenge text files."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(".")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(
        ROOT / "outputs/n22/live_cal10_proto"))
    parser.add_argument("--out", default="")
    parser.add_argument("--tracker", default="N22_PROTO")
    args = parser.parse_args()
    source_root = Path(args.source)
    out_root = Path(args.out) if args.out else source_root / "trackeval"
    tracker = out_root / "mot_results" / args.tracker
    tracker.mkdir(parents=True, exist_ok=True)
    sequences = []
    manifest = {}
    for source in sorted(source_root.glob("events_*.jsonl")):
        sequence = source.stem.removeprefix("events_")
        rows = []
        seen = set()
        skipped = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            box = event.get("delivered_box")
            if not event.get("delivered") or box is None:
                skipped += 1
                continue
            if len(box) != 4:
                skipped += 1
                continue
            frame = int(event["frame"]) + 1
            track_id = int(event["public_id"])
            key = (frame, track_id)
            if key in seen:
                raise RuntimeError(f"duplicate output key {sequence} {key}")
            seen.add(key)
            x1, y1, x2, y2 = [float(value) for value in box]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                skipped += 1
                continue
            score = event.get("delivery_score")
            score = 1.0 if score is None else float(score)
            rows.append((frame, track_id, x1, y1, width, height, score))
        rows.sort(key=lambda row: (row[0], row[1]))
        target = tracker / f"{sequence}.txt"
        target.write_text(
            "".join(
                f"{frame},{track_id},{x1:.2f},{y1:.2f},{width:.2f},"
                f"{height:.2f},{score:.3f},-1,-1,-1\n"
                for frame, track_id, x1, y1, width, height, score in rows
            ),
            encoding="utf-8",
        )
        sequences.append(sequence)
        manifest[sequence] = {
            "event_rows": len(rows) + skipped,
            "mot_rows": len(rows),
            "skipped_rows": skipped,
        }
    (out_root / "seqmap.txt").write_text(
        "name\n" + "\n".join(sequences) + "\n", encoding="utf-8")
    (out_root / "export_manifest.json").write_text(
        json.dumps({"tracker": args.tracker, "sequences": manifest},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"tracker": args.tracker, "sequences": sequences,
                      "manifest": manifest},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
