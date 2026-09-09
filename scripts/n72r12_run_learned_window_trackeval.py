#!/usr/bin/env python3
"""Run pinned official TrackEval for the 32-event four-variant E1D matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.window_trackeval import write_json_atomic, write_text_atomic  # noqa: E402

HORIZONS = (20, 50, 100)
VARIANTS = ("E0_BASELINE_B0", "E1B_PCTIS", "E1C_PCTIS_SAFE", "E1D_PCTIS_LEARNED_SAFE")
METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDSW", "IDF1")
EXPECTED_EVENTS = 32


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _number(value: str, label: str) -> float:
    result = float(value)
    if not (result == result and result not in (float("inf"), float("-inf"))):
        raise RuntimeError(f"non-finite TrackEval value {label}")
    return result


def _detailed(path: Path, expected: set[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not path.is_file():
        raise RuntimeError(f"missing TrackEval detailed CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = {"HOTA___AUC", "DetA___AUC", "AssA___AUC", "MOTA", "IDSW", "IDF1"}
    if not rows or fields - set(rows[0]):
        raise RuntimeError(f"TrackEval CSV schema is incomplete: {path}")
    combined: dict[str, float] | None = None
    per_sequence: dict[str, dict[str, float]] = {}
    for row in rows:
        seq = row.get("seq", "")
        if not seq:
            continue
        metrics = {
            field.removesuffix("___AUC") if field.endswith("___AUC") else field: _number(row[field], f"{path}/{seq}/{field}")
            for field in fields
        }
        if seq == "COMBINED":
            combined = metrics
        else:
            per_sequence[seq] = metrics
    if combined is None or set(per_sequence) != expected:
        raise RuntimeError(f"TrackEval sequence rows incomplete: {path}")
    return combined, per_sequence


def _command(entry: Path, trackeval: Path, export: Path, raw: Path, seqmap: Path) -> list[str]:
    return [
        sys.executable, str(entry), str(trackeval),
        "--GT_FOLDER", str(export / "pseudo_gt"),
        "--TRACKERS_FOLDER", str(export / "trackers"),
        "--TRACKERS_TO_EVAL", *VARIANTS,
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_FOLDER", str(raw),
        "--OUTPUT_SUB_FOLDER", "",
        "--SEQ_INFO", *[line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines()[1:] if line.strip()],
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


def horizon(export: Path, output: Path, trackeval: Path, value: int) -> dict[str, Any]:
    manifest = _read(export / "export_manifest.json")
    windows = {item["pseudo_sequence"]: item for item in manifest["event_windows"] if int(item["horizon"]) == value}
    if len(windows) != EXPECTED_EVENTS:
        raise RuntimeError(f"H{value}: expected {EXPECTED_EVENTS} windows")
    raw = output / "trackeval_raw" / f"h{value:03d}"
    seqmap = export / "seqmaps" / f"seqmap_h{value:03d}.txt"
    command = _command(ROOT / "scripts/n72r11r5_trackeval_entry.py", trackeval / "scripts/run_mot_challenge.py", export, raw, seqmap)
    process = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    log = raw / "trackeval.log"
    write_text_atomic(log, process.stdout + process.stderr)
    result: dict[str, Any] = {"horizon": value, "command": command, "returncode": process.returncode, "log_path": str(log), "status": "PASS" if process.returncode == 0 else "ERROR"}
    if process.returncode != 0:
        result["stdout_tail"] = process.stdout[-5000:]
        result["stderr_tail"] = process.stderr[-5000:]
        write_json_atomic(raw / "run_result.json", result)
        return result
    records: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, float]] = {}
    for logical in VARIANTS:
        combined, per_sequence = _detailed(raw / logical / "pedestrian_detailed.csv", set(windows))
        pooled[logical] = combined
        for pseudo, metrics in sorted(per_sequence.items()):
            window = windows[pseudo]
            records.append({
                "logical_variant": logical,
                "pseudo_sequence": pseudo,
                "original_sequence": window["original_sequence"],
                "event_id": window["event_id"],
                "action_type": window["action_type"],
                "event_frame": window["event_frame"],
                "horizon": value,
                "metrics": metrics,
                "trackeval_detail_path": str(raw / logical / "pedestrian_detailed.csv"),
            })
    expected = EXPECTED_EVENTS * len(VARIANTS)
    keys = [(row["event_id"], row["logical_variant"], row["horizon"]) for row in records]
    if len(records) != expected or len(set(keys)) != expected:
        raise RuntimeError(f"H{value}: record completeness failed")
    result.update({"record_count": len(records), "pooled": pooled, "records": records})
    write_json_atomic(raw / "run_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=ROOT / "outputs/N72R12/trackeval_learned")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R12/trackeval_learned")
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    export = args.export_root.resolve()
    output = args.output_root.resolve()
    trackeval = args.trackeval_root.resolve()
    try:
        export_manifest = _read(export / "export_manifest.json")
        if export_manifest.get("status") != "PASS_EXPORT_N72R12_LEARNED" or export_manifest.get("record_count") != 384:
            raise RuntimeError("learned TrackEval export is incomplete")
        if tuple(sorted(args.horizons)) != HORIZONS:
            raise RuntimeError(f"TrackEval must run exactly H20/H50/H100: {args.horizons}")
        results = [horizon(export, output, trackeval, value) for value in HORIZONS]
        failed = [item for item in results if item.get("status") != "PASS"]
        rows = [row for item in results if item.get("status") == "PASS" for row in item.get("records", [])]
        keys = [(row["event_id"], row["logical_variant"], row["horizon"]) for row in rows]
        payload = {
            "schema_version": "N72R12_LEARNED_TRACKEVAL_RUN_MANIFEST_V1",
            "status": "PASS_TRACKEVAL_N72R12_LEARNED" if not failed and len(rows) == 384 and len(set(keys)) == 384 else "FAIL_TRACKEVAL_N72R12_LEARNED",
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "horizons": list(HORIZONS),
            "logical_variants": list(VARIANTS),
            "expected_record_count": 384,
            "record_count": len(rows),
            "duplicate_keys": sorted({key for key in keys if keys.count(key) > 1}),
            "horizon_results": results,
            "cuda_visible_devices": "",
            "runtime_future_gt_used": False,
            "gt_used_only_for_posthoc_evaluation": True,
            "trackeval_commit": export_manifest.get("trackeval_commit"),
            "trackeval_root": export_manifest.get("trackeval_root"),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        output.mkdir(parents=True, exist_ok=True)
        if payload["status"] == "PASS_TRACKEVAL_N72R12_LEARNED":
            write_text_atomic(output / "per_event_metrics.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        else:
            write_text_atomic(output / "per_event_metrics_partial.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        write_json_atomic(output / "trackeval_run_manifest.json", payload)
        print(json.dumps({"status": payload["status"], "records": len(rows), "expected": 384, "output": str(output / "trackeval_run_manifest.json")}, sort_keys=True))
        return 0 if payload["status"] == "PASS_TRACKEVAL_N72R12_LEARNED" else 1
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "trackeval_failure.json", {"schema_version": "N72R12_LEARNED_TRACKEVAL_FAILURE_V1", "status": "FAIL_TRACKEVAL_N72R12_LEARNED", "error_type": type(exc).__name__, "error": str(exc), "traceback": __import__("traceback").format_exc(), "historical_outputs_modified": False})
        print(json.dumps({"status": "FAIL_TRACKEVAL_N72R12_LEARNED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
