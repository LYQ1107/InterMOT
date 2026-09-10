#!/usr/bin/env python3
"""Run pinned official TrackEval on the sealed N72R14 interaction windows."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.window_trackeval import (  # noqa: E402
    WindowTrackEvalError,
    write_json_atomic,
    write_text_atomic,
)


HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1F_TARGET_POOL_ONLY",
    "E1G_TARGET_STATE_ONLY",
    "E1H_PERSISTENT_GLOBAL_STATE",
)
METRIC_FIELDS = ("HOTA___AUC", "DetA___AUC", "AssA___AUC", "MOTA", "IDSW", "IDF1")
PINNED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _finite(value: str, label: str) -> float:
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise WindowTrackEvalError(f"{label} is non-finite")
    return result


def _detailed(path: Path, expected_sequences: set[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise WindowTrackEvalError(f"missing TrackEval detailed CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise WindowTrackEvalError(f"empty TrackEval detailed CSV: {path}")
    missing = sorted(set(METRIC_FIELDS) - set(rows[0]))
    if missing:
        raise WindowTrackEvalError(f"TrackEval CSV missing {missing}: {path}")
    combined: dict[str, float] | None = None
    per_sequence: dict[str, dict[str, float]] = {}
    for row in rows:
        sequence = row.get("seq", "")
        if not sequence:
            continue
        metrics = {
            field.removesuffix("___AUC") if field.endswith("___AUC") else field:
            _finite(row[field], f"{path}:{sequence}:{field}")
            for field in METRIC_FIELDS
        }
        if sequence == "COMBINED":
            combined = metrics
        else:
            per_sequence[sequence] = metrics
    if combined is None or set(per_sequence) != expected_sequences:
        raise WindowTrackEvalError(
            f"TrackEval rows incomplete for {path}: "
            f"missing={sorted(expected_sequences - set(per_sequence))[:4]}"
        )
    return combined, per_sequence


def _command(export: Path, raw: Path, seqmap: Path, trackeval: Path) -> list[str]:
    entry = ROOT / "scripts/n72r11r5_trackeval_entry.py"
    names = [line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    if not names:
        raise WindowTrackEvalError(f"empty seqmap: {seqmap}")
    return [
        sys.executable,
        str(entry),
        str(trackeval / "scripts" / "run_mot_challenge.py"),
        "--GT_FOLDER",
        str(export / "pseudo_gt"),
        "--TRACKERS_FOLDER",
        str(export / "trackers"),
        "--TRACKERS_TO_EVAL",
        *VARIANTS,
        "--TRACKER_SUB_FOLDER",
        "data",
        "--OUTPUT_FOLDER",
        str(raw),
        "--OUTPUT_SUB_FOLDER",
        "",
        "--SEQ_INFO",
        *sorted(names),
        "--BENCHMARK",
        "InterMOTWindow",
        "--SPLIT_TO_EVAL",
        "all",
        "--SKIP_SPLIT_FOL",
        "True",
        "--DO_PREPROC",
        "False",
        "--CLASSES_TO_EVAL",
        "pedestrian",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--USE_PARALLEL",
        "False",
        "--NUM_PARALLEL_CORES",
        "1",
        "--BREAK_ON_ERROR",
        "True",
        "--PRINT_RESULTS",
        "False",
        "--PRINT_ONLY_COMBINED",
        "False",
        "--PRINT_CONFIG",
        "False",
        "--TIME_PROGRESS",
        "False",
        "--OUTPUT_SUMMARY",
        "True",
        "--OUTPUT_DETAILED",
        "True",
        "--OUTPUT_EMPTY_CLASSES",
        "True",
        "--PLOT_CURVES",
        "False",
    ]


def _validate_export(export: Path, trackeval: Path) -> dict[str, Any]:
    manifest = _read(export / "export_manifest.json")
    if manifest.get("status") != "PASS_EXPORT_N72R14":
        raise WindowTrackEvalError(f"N72R14 export is not PASS: {manifest.get('status')}")
    if int(manifest.get("source_events", -1)) != 32 or int(manifest.get("record_count", -1)) != 384:
        raise WindowTrackEvalError("N72R14 export does not contain 32 events and 384 records")
    if manifest.get("runtime_future_gt_used") is not False:
        raise WindowTrackEvalError("export manifest has runtime future GT")
    commit = subprocess.check_output(["git", "-C", str(trackeval), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_TRACKEVAL_COMMIT:
        raise WindowTrackEvalError(f"TrackEval commit {commit} != pinned {PINNED_TRACKEVAL_COMMIT}")
    return manifest


def run_horizon(export: Path, output: Path, trackeval: Path, horizon: int) -> dict[str, Any]:
    manifest = _read(export / "export_manifest.json")
    windows = {
        str(item["pseudo_sequence"]): item
        for item in manifest.get("event_windows", [])
        if int(item.get("horizon", -1)) == int(horizon)
    }
    if len(windows) != 32:
        raise WindowTrackEvalError(f"H{horizon}: expected 32 event windows, got {len(windows)}")
    seqmap = export / "seqmaps" / f"seqmap_h{horizon:03d}.txt"
    raw = output / "trackeval_raw" / f"h{horizon:03d}"
    command = _command(export, raw, seqmap, trackeval)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    log = raw / "trackeval.log"
    write_text_atomic(log, process.stdout + process.stderr)
    result: dict[str, Any] = {
        "horizon": int(horizon),
        "command": command,
        "returncode": int(process.returncode),
        "log_path": str(log),
        "status": "PASS" if process.returncode == 0 else "ERROR",
        "cuda_visible_devices": "",
    }
    if process.returncode != 0:
        result["stdout_tail"] = process.stdout[-5000:]
        result["stderr_tail"] = process.stderr[-5000:]
        write_json_atomic(raw / "run_result.json", result)
        return result
    pooled: dict[str, dict[str, float]] = {}
    records: list[dict[str, Any]] = []
    expected_sequences = set(windows)
    for variant in VARIANTS:
        detail = raw / variant / "pedestrian_detailed.csv"
        combined, per_sequence = _detailed(detail, expected_sequences)
        pooled[variant] = combined
        for pseudo, metrics in sorted(per_sequence.items()):
            window = windows[pseudo]
            records.append(
                {
                    "logical_variant": variant,
                    "pseudo_sequence": pseudo,
                    "original_sequence": window["original_sequence"],
                    "event_id": window["event_id"],
                    "action_type": window["action_type"],
                    "event_frame": window["event_frame"],
                    "horizon": int(horizon),
                    "metrics": metrics,
                    "trackeval_detail_path": str(detail),
                }
            )
    expected = 32 * len(VARIANTS)
    keys = [(row["event_id"], row["logical_variant"], int(row["horizon"])) for row in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise WindowTrackEvalError(f"H{horizon}: per-event key completeness failed")
    result.update({"record_count": len(records), "pooled": pooled, "records": records})
    write_json_atomic(raw / "run_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=ROOT / "outputs/N72R14/trackeval")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R14/trackeval")
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    export = args.export_root.resolve()
    output = args.output_root.resolve()
    trackeval = args.trackeval_root.resolve()
    stage_status = output.parent / "stage_06_status.json"
    try:
        if tuple(sorted(args.horizons)) != HORIZONS:
            raise WindowTrackEvalError(f"must run exactly H20/H50/H100: {args.horizons}")
        export_manifest = _validate_export(export, trackeval)
        results = [run_horizon(export, output, trackeval, horizon) for horizon in HORIZONS]
        rows = [row for result in results if result.get("status") == "PASS" for row in result.get("records", [])]
        expected = 32 * len(HORIZONS) * len(VARIANTS)
        keys = [(row["event_id"], row["logical_variant"], int(row["horizon"])) for row in rows]
        duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
        failed = [result for result in results if result.get("status") != "PASS"]
        status = "PASS_TRACKEVAL_N72R14" if not failed and len(rows) == expected and not duplicate_keys else "BLOCKED_INCOMPLETE_TRACKEVAL_N72R14"
        manifest = {
            "schema_version": "N72R14_TRACKEVAL_RUN_MANIFEST_V1",
            "status": status,
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "horizons": list(HORIZONS),
            "logical_variants": list(VARIANTS),
            "expected_record_count": expected,
            "record_count": len(rows),
            "duplicate_keys": duplicate_keys,
            "missing_keys": sorted(
                {
                    (event["event_id"], variant, horizon)
                    for event in export_manifest.get("event_windows", [])
                    if int(event.get("horizon", -1)) == 100
                    for variant in VARIANTS
                    for horizon in HORIZONS
                }
                - set(keys)
            ),
            "horizon_results": results,
            "cuda_visible_devices": "",
            "runtime_future_gt_used": False,
            "gt_used_only_for_posthoc_evaluation": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "trackeval_commit": PINNED_TRACKEVAL_COMMIT,
            "export_manifest": str(export / "export_manifest.json"),
            "export_manifest_sha256": hashlib.sha256((export / "export_manifest.json").read_bytes()).hexdigest(),
            "created_at_utc": now_utc(),
        }
        write_json_atomic(output / "trackeval_run_manifest.json", manifest)
        write_text_atomic(
            output / ("per_event_metrics.jsonl" if status == "PASS_TRACKEVAL_N72R14" else "per_event_metrics_partial.jsonl"),
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
        stage = {
            "schema_version": "N72R14_STAGE_STATUS_V1",
            "stage": "N72R14-06-OFFICIAL-TRACKEVAL",
            "status": status,
            "manifest": str(output / "trackeval_run_manifest.json"),
            "record_count": len(rows),
            "expected_record_count": expected,
            "duplicate_key_count": len(duplicate_keys),
            "horizon_statuses": {str(result["horizon"]): result["status"] for result in results},
            "trackeval_commit": PINNED_TRACKEVAL_COMMIT,
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        write_json_atomic(stage_status, stage)
        print(json.dumps(stage, sort_keys=True))
        return 0 if status == "PASS_TRACKEVAL_N72R14" else 1
    except Exception as exc:
        failure = {
            "schema_version": "N72R14_FAILURE_V1",
            "status": "FAIL_N72R14_TRACKEVAL_RUNNER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        write_json_atomic(output / "trackeval_failure.json", failure)
        write_json_atomic(stage_status, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
