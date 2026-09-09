#!/usr/bin/env python
"""Export sealed N72R11R4 future windows to TrackEval's MOT layout."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import WindowTrackEvalError, export_windows, write_json_atomic


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _write_exception_artifact(output_root: Path, exc: BaseException) -> Path:
    """Persist an exporter exception without replacing an earlier attempt."""
    attempt = 1
    while True:
        path = output_root / f"export_failure_attempt{attempt}.json"
        if not path.exists():
            break
        attempt += 1
    write_json_atomic(
        path,
        {
            "schema_version": "N72R11R5_EXPORT_EXCEPTION_V1",
            "status": "FAIL_EXPORT_EXCEPTION",
            "command": [sys.executable, *sys.argv],
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "output_root": str(output_root),
            "trackeval_started": False,
            "historical_runtime_artifacts_modified": False,
        },
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs/N72R11R5"))
    parser.add_argument("--data-root", type=Path, default=Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"))
    parser.add_argument("--trackeval-root", type=Path, default=Path("third_party/MOTIP/TrackEval"))
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--e1a-metrics",
        type=Path,
        default=None,
        help="optional explicit E1A metrics source; pair with --e1b-metrics for corrected replay",
    )
    parser.add_argument(
        "--e1b-metrics",
        type=Path,
        default=None,
        help="optional explicit E1B metrics source; pair with --e1a-metrics for corrected replay",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = (project_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    trackeval_root = (project_root / args.trackeval_root).resolve() if not args.trackeval_root.is_absolute() else args.trackeval_root.resolve()
    if (args.e1a_metrics is None) != (args.e1b_metrics is None):
        parser.error("--e1a-metrics and --e1b-metrics must be supplied together")
    metrics_paths = None
    if args.e1a_metrics is not None:
        metrics_paths = {
            "E1A_V3": str(args.e1a_metrics),
            "E1B_PCTIS": str(args.e1b_metrics),
        }
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
            metrics_paths=metrics_paths,
        )
    except (WindowTrackEvalError, subprocess.CalledProcessError, OSError, KeyError, TypeError, ValueError) as exc:
        failure_path = _write_exception_artifact(output_root, exc)
        print(f"EXPORT_ERROR: {exc}; failure_artifact={failure_path}")
        return 1
    print(
        f"EXPORT_PASS events={manifest['source_events']} windows={manifest['pseudo_sequence_count']} "
        f"records={manifest['record_count']} output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
