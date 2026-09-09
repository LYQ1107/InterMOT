#!/usr/bin/env python
"""Export sealed N72R11R4 future windows to TrackEval's MOT layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import WindowTrackEvalError, export_windows


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs/N72R11R5"))
    parser.add_argument("--data-root", type=Path, default=Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"))
    parser.add_argument("--trackeval-root", type=Path, default=Path("third_party/MOTIP/TrackEval"))
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = (project_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    trackeval_root = (project_root / args.trackeval_root).resolve() if not args.trackeval_root.is_absolute() else args.trackeval_root.resolve()
    try:
        manifest = export_windows(
            root=output_root,
            data_root=data_root,
            source_branch=_git_value(project_root, "symbolic-ref", "--short", "HEAD"),
            source_commit=_git_value(project_root, "rev-parse", "HEAD"),
            trackeval_root=trackeval_root,
            event_ids=args.event_ids,
            mode=args.mode,
            project_root=project_root,
        )
    except (WindowTrackEvalError, subprocess.CalledProcessError) as exc:
        print(f"EXPORT_ERROR: {exc}")
        return 1
    print(
        f"EXPORT_PASS events={manifest['source_events']} windows={manifest['pseudo_sequence_count']} "
        f"records={manifest['record_count']} output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
