#!/usr/bin/env python
"""Run the pinned official TrackEval evaluator on exported windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import (
    HORIZONS,
    LOGICAL_VARIANTS,
    WindowTrackEvalError,
    write_json_atomic,
    write_text_atomic,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _parse_number(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise WindowTrackEvalError(f"{label} is not numeric: {value!r}") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise WindowTrackEvalError(f"{label} is non-finite")
    return result


def _load_detailed(path: Path, expected_sequences: set[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise WindowTrackEvalError(f"TrackEval detailed output missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise WindowTrackEvalError(f"TrackEval detailed output is empty: {path}")
    required = {"HOTA___AUC", "DetA___AUC", "AssA___AUC", "MOTA", "IDSW", "IDF1"}
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise WindowTrackEvalError(f"TrackEval output lacks required fields {missing_columns}: {path}")
    combined: dict[str, float] | None = None
    per_sequence: dict[str, dict[str, float]] = {}
    for row in rows:
        sequence = row.get("seq", "")
        if not sequence:
            continue
        metrics = {
            field.removesuffix("___AUC") if field.endswith("___AUC") else field: _parse_number(
                row[field], f"{path}:{sequence}:{field}"
            )
            for field in required
        }
        if sequence == "COMBINED":
            combined = metrics
        else:
            per_sequence[sequence] = metrics
    if combined is None:
        raise WindowTrackEvalError(f"TrackEval combined row missing: {path}")
    if set(per_sequence) != expected_sequences:
        raise WindowTrackEvalError(
            f"TrackEval sequence rows differ from seqmap for {path}: "
            f"missing={sorted(expected_sequences - set(per_sequence))[:5]}, "
            f"unexpected={sorted(set(per_sequence) - expected_sequences)[:5]}"
        )
    return combined, per_sequence


def _trackeval_command(
    *, python: str, entry: Path, trackeval_script: Path, export_root: Path, raw_root: Path,
    seqmap: Path,
) -> list[str]:
    return [
        python,
        str(entry),
        str(trackeval_script),
        "--GT_FOLDER", str(export_root / "pseudo_gt"),
        "--TRACKERS_FOLDER", str(export_root / "trackers"),
        "--TRACKERS_TO_EVAL", *LOGICAL_VARIANTS,
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_FOLDER", str(raw_root),
        "--OUTPUT_SUB_FOLDER", "",
        # This pinned CLI parses SEQMAP_FILE (whose default is None) as a
        # one-element list.  SEQ_INFO is the supported direct-sequence path in
        # the same official dataset adapter and reads the exported seqinfo.ini.
        "--SEQ_INFO", *sorted(
            line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines()[1:] if line.strip()
        ),
        "--BENCHMARK", "InterMOTWindow",
        "--SPLIT_TO_EVAL", "all",
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "False",
        "--CLASSES_TO_EVAL", "pedestrian",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1",
        "--BREAK_ON_ERROR", "True",
        "--PRINT_RESULTS", "False",
        "--PRINT_ONLY_COMBINED", "False",
        "--PRINT_CONFIG", "False",
        "--TIME_PROGRESS", "False",
        "--OUTPUT_SUMMARY", "True",
        "--OUTPUT_DETAILED", "True",
        "--OUTPUT_EMPTY_CLASSES", "True",
        "--PLOT_CURVES", "False",
    ]


def run_horizon(
    *, export_root: Path, output_root: Path, horizon: int, python: str, trackeval_root: Path,
) -> dict[str, Any]:
    export_manifest = _read_json(export_root / "export_manifest.json")
    event_windows = {
        item["pseudo_sequence"]: item
        for item in export_manifest.get("event_windows", [])
        if item.get("horizon") == horizon
    }
    if len(event_windows) != export_manifest.get("source_events"):
        raise WindowTrackEvalError(f"H{horizon}: export manifest does not contain every pseudo sequence")
    seqmap = export_root / "seqmaps" / f"seqmap_h{horizon:03d}.txt"
    raw_root = output_root / "trackeval_raw" / f"h{horizon:03d}"
    log_path = raw_root / "trackeval.log"
    entry = Path(__file__).resolve().with_name("n72r11r5_trackeval_entry.py")
    trackeval_script = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not entry.is_file() or not trackeval_script.is_file():
        raise WindowTrackEvalError(f"official TrackEval entry is missing: {trackeval_script}")
    command = _trackeval_command(
        python=python, entry=entry, trackeval_script=trackeval_script,
        export_root=export_root, raw_root=raw_root, seqmap=seqmap,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    write_text_atomic(log_path, process.stdout + process.stderr)
    result: dict[str, Any] = {
        "horizon": horizon,
        "command": command,
        "returncode": process.returncode,
        "log_path": str(log_path),
        "status": "PASS" if process.returncode == 0 else "ERROR",
    }
    if process.returncode != 0:
        result["stderr_tail"] = process.stderr[-4000:]
        result["stdout_tail"] = process.stdout[-4000:]
        write_json_atomic(raw_root / "run_result.json", result)
        return result

    records: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, float]] = {}
    for logical in LOGICAL_VARIANTS:
        detailed_path = raw_root / logical / "pedestrian_detailed.csv"
        combined, per_sequence = _load_detailed(detailed_path, set(event_windows))
        pooled[logical] = combined
        for pseudo, metrics in sorted(per_sequence.items()):
            window = event_windows[pseudo]
            records.append(
                {
                    "logical_variant": logical,
                    "pseudo_sequence": pseudo,
                    "original_sequence": window["original_sequence"],
                    "event_id": window["event_id"],
                    "action_type": window["action_type"],
                    "event_frame": window["event_frame"],
                    "horizon": horizon,
                    "metrics": metrics,
                    "trackeval_detail_path": str(detailed_path),
                }
            )
    expected = len(event_windows) * len(LOGICAL_VARIANTS)
    keys = [(r["event_id"], r["logical_variant"], r["horizon"]) for r in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise WindowTrackEvalError(f"H{horizon}: per-event TrackEval record completeness failed")
    result.update({"status": "PASS", "record_count": len(records), "pooled": pooled, "records": records})
    write_json_atomic(raw_root / "run_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=Path("outputs/N72R11R5"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--trackeval-root", type=Path, default=Path("third_party/MOTIP/TrackEval"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    export_root = args.export_root.resolve()
    output_root = (args.output_root or export_root).resolve()
    trackeval_root = args.trackeval_root.resolve()
    try:
        manifest = _read_json(export_root / "export_manifest.json")
        if manifest.get("status") != "PASS_EXPORT":
            raise WindowTrackEvalError("cannot run TrackEval before PASS_EXPORT")
        if sorted(args.horizons) != sorted(set(args.horizons)) or not set(args.horizons).issubset(HORIZONS):
            raise WindowTrackEvalError(f"invalid horizons: {args.horizons}")
        horizon_results = []
        all_records: list[dict[str, Any]] = []
        failed = False
        for horizon in args.horizons:
            try:
                result = run_horizon(
                    export_root=export_root, output_root=output_root, horizon=horizon,
                    python=args.python, trackeval_root=trackeval_root,
                )
            except WindowTrackEvalError as exc:
                result = {"horizon": horizon, "status": "ERROR", "reason": str(exc)}
            horizon_results.append(result)
            if result.get("status") != "PASS":
                failed = True
            else:
                all_records.extend(result.get("records", []))
        expected = manifest["source_events"] * len(LOGICAL_VARIANTS) * len(args.horizons)
        keys = [(r["event_id"], r["logical_variant"], r["horizon"]) for r in all_records]
        run_manifest = {
            "schema_version": "N72R11R5_TRACKEVAL_RUN_MANIFEST_V1",
            "status": "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL" if failed or len(keys) != expected else "PASS_TRACKEVAL",
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "cuda_visible_devices": "",
            "horizons": args.horizons,
            "logical_variants": list(LOGICAL_VARIANTS),
            "expected_record_count": expected,
            "record_count": len(all_records),
            "duplicate_keys": sorted({key for key in keys if keys.count(key) > 1}),
            "horizon_results": horizon_results,
        }
        write_json_atomic(output_root / "trackeval_run_manifest.json", run_manifest)
        if not failed and len(keys) == expected:
            write_text_atomic(
                output_root / "per_event_metrics.jsonl",
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records),
            )
        else:
            write_text_atomic(
                output_root / "per_event_metrics_partial.jsonl",
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records),
            )
        print(f"TRACKEVAL_{'PASS' if run_manifest['status'] == 'PASS_TRACKEVAL' else 'INCOMPLETE'} records={len(all_records)} expected={expected}")
        return 0 if run_manifest["status"] == "PASS_TRACKEVAL" else 1
    except (OSError, KeyError, TypeError, WindowTrackEvalError) as exc:
        print(f"TRACKEVAL_ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
