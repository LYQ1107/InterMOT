#!/usr/bin/env python3
"""Export the frozen N72R13 Oracle comparison to the pinned TrackEval layout.

This exporter deliberately compares the historical frozen E0/E1B sealed streams
with the new sequential Oracle stream.  The Oracle stream is a causal runtime
artifact; GT is opened only while constructing the post-hoc pseudo windows.
No historical N72R11R5R1 or N72R13 runtime file is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation import window_trackeval as wt  # noqa: E402

HORIZONS = (20, 50, 100)
VARIANTS = ("E0_BASELINE_B0", "E1B_PCTIS", "ORACLE_TIV")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise wt.WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_jsonl(path: Path, event_id: str, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise wt.WindowTrackEvalError(f"{event_id}/{variant}: non-object line {line_number}")
            rows.append(value)
    if len(rows) != 101:
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: expected 101 rows, got {len(rows)}")
    event_frame = rows[0].get("event_frame")
    if not isinstance(event_frame, int):
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: missing integer event_frame")
    if [row.get("frame") for row in rows] != list(range(event_frame, event_frame + 101)):
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: incomplete event..event+100 frame axis")
    for row in rows:
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            raise wt.WindowTrackEvalError(f"{event_id}/{variant}: runtime GT flag is not false")
        if row.get("posthoc_gt_used") is not False:
            raise wt.WindowTrackEvalError(f"{event_id}/{variant}: runtime row contains posthoc GT")
    event_row = rows[0]
    if event_row.get("event_frame_memory_read") is not False or event_row.get("memory_read") is not False:
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: event frame reads memory")
    if rows[1].get("frame") != event_frame + 1:
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: event+1 is missing")
    return rows


def _frozen_runtime_paths(r5: dict[str, Any]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for record in r5.get("records", []):
        event_id = str(record["event_id"])
        done = Path(str(record["done"]))
        event_root = done.parent
        result[event_id] = {
            "E0_BASELINE_B0": event_root / "E0_BASELINE_B0" / "runtime_frames.jsonl",
            "E1B_PCTIS": event_root / "E1B_PCTIS_LEGACY" / "runtime_frames.jsonl",
        }
    if len(result) != 32:
        raise wt.WindowTrackEvalError(f"frozen E1B manifest does not contain 32 unique events: {len(result)}")
    return result


def _oracle_runtime_path(event_root: Path, event_id: str, output_root: Path) -> Path:
    sealed = _read(event_root / "runtime_event_sealed.json")
    if sealed.get("runtime_future_gt_used") is not False or sealed.get("runtime_gt_read") is not False:
        raise wt.WindowTrackEvalError(f"{event_id}: Oracle sealed artifact has runtime GT usage")
    if sealed.get("oracle_upper_bound_only") is not True or sealed.get("runtime_eligible") is not False:
        raise wt.WindowTrackEvalError(f"{event_id}: Oracle provenance flags are inconsistent")
    source_rows = sealed.get("oracle_rows")
    if not isinstance(source_rows, list):
        raise wt.WindowTrackEvalError(f"{event_id}: Oracle rows are missing")
    rows = [row for row in source_rows if isinstance(row, dict)]
    if len(rows) != len(source_rows):
        raise wt.WindowTrackEvalError(f"{event_id}: Oracle rows contain a non-object")
    destination = output_root / "runtime_sources" / "ORACLE_TIV" / event_id / "runtime_frames.jsonl"
    _atomic_jsonl(destination, rows)
    return destination


def export(output_root: Path, data_root: Path, protocol_path: Path, r5_manifest_path: Path,
           oracle_root: Path, trackeval_root: Path) -> dict[str, Any]:
    r5 = _read(r5_manifest_path)
    if r5.get("status") != "PASS_ALL_SELECTED" or r5.get("event_count") != 32:
        raise wt.WindowTrackEvalError("frozen E1B source manifest is incomplete")
    protocol = _read(protocol_path)
    events = {
        str(item["event_id"]): dict(item)
        for item in protocol.get("source_event_selection", {}).get("events", [])
        if isinstance(item, dict) and item.get("event_id") is not None
    }
    frozen_paths = _frozen_runtime_paths(r5)
    if len(events) != 32 or set(events) != set(frozen_paths):
        raise wt.WindowTrackEvalError("N72R13 event set is not exactly the frozen 32-event protocol")
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_event: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_records: list[dict[str, Any]] = []
    for event_id in sorted(events):
        event_root = oracle_root / "events" / event_id
        oracle_path = _oracle_runtime_path(event_root, event_id, output_root)
        paths = {**frozen_paths[event_id], "ORACLE_TIV": oracle_path}
        rows_by_event[event_id] = {}
        for variant, path in paths.items():
            if not path.is_file():
                raise wt.WindowTrackEvalError(f"{event_id}/{variant}: missing runtime source {path}")
            rows_by_event[event_id][variant] = _load_jsonl(path, event_id, variant)
            source_records.append({"event_id": event_id, "variant": variant, "path": str(path), "sha256": _sha(path)})

    windows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seqmaps: dict[int, list[str]] = {horizon: [] for horizon in HORIZONS}
    for event_id in sorted(events):
        event = events[event_id]
        event_frame = int(event["event_frame"])
        for horizon in HORIZONS:
            pseudo = wt._pseudo_name(event_id, horizon)
            resolved = wt.resolve_source_gt_and_seqinfo(data_root, {**event, "frozen_event": event}, project_root=ROOT)
            gt_source = Path(resolved["gt_path"])
            seqinfo_source = Path(resolved["seqinfo_path"])
            gt_lines = wt._gt_window_lines(gt_source, event_frame=event_frame, horizon=horizon, event_id=event_id)
            gt_dir = output_root / "pseudo_gt" / pseudo
            wt.write_text_atomic(gt_dir / "gt" / "gt.txt", "\n".join(gt_lines) + "\n")
            wt.write_text_atomic(gt_dir / "seqinfo.ini", wt._seqinfo_text(seqinfo_source, pseudo, horizon))
            seqmaps[horizon].append(pseudo)
            windows.append({
                "pseudo_sequence": pseudo,
                "original_sequence": event["sequence"],
                "event_id": event_id,
                "action_type": event["action_type"],
                "event_frame": event_frame,
                "horizon": horizon,
                "original_start_frame": event_frame + 1,
                "original_end_frame": event_frame + horizon,
                "gt_path": str(gt_dir / "gt" / "gt.txt"),
                "seqinfo_path": str(gt_dir / "seqinfo.ini"),
                "gt_source_path": str(gt_source),
                "seqinfo_source_path": str(seqinfo_source),
                "gt_source_resolution": resolved["resolution"],
                "gt_sha256": _sha(gt_dir / "gt" / "gt.txt"),
            })
            for variant in VARIANTS:
                prediction_lines, counts = wt._prediction_lines(
                    rows_by_event[event_id][variant], event_frame=event_frame, horizon=horizon,
                    event_id=event_id, variant=variant,
                )
                prediction_path = output_root / "trackers" / variant / "data" / f"{pseudo}.txt"
                wt.write_text_atomic(prediction_path, "\n".join(prediction_lines) + ("\n" if prediction_lines else ""))
                records.append({
                    "logical_variant": variant,
                    "runtime_frames_path": source_records[[
                        (item["event_id"], item["variant"]) for item in source_records
                    ].index((event_id, variant))]["path"],
                    "runtime_frames_sha256": source_records[[
                        (item["event_id"], item["variant"]) for item in source_records
                    ].index((event_id, variant))]["sha256"],
                    "pseudo_sequence": pseudo,
                    "original_sequence": event["sequence"],
                    "event_id": event_id,
                    "action_type": event["action_type"],
                    "event_frame": event_frame,
                    "horizon": horizon,
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": _sha(prediction_path),
                    **counts,
                })
    for horizon, names in seqmaps.items():
        wt.write_text_atomic(output_root / "seqmaps" / f"seqmap_h{horizon:03d}.txt", "name\n" + "\n".join(names) + "\n")
    expected = 32 * len(HORIZONS) * len(VARIANTS)
    keys = [(item["event_id"], item["logical_variant"], int(item["horizon"])) for item in records]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    expected_keys = {(event_id, variant, horizon) for event_id in events for variant in VARIANTS for horizon in HORIZONS}
    missing = sorted(expected_keys - set(keys))
    if len(records) != expected or duplicates or missing:
        raise wt.WindowTrackEvalError(f"N72R13 export incomplete records={len(records)} duplicates={duplicates} missing={missing[:5]}")
    try:
        trackeval_commit = subprocess.check_output(["git", "-C", str(trackeval_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        trackeval_commit = None
    manifest = {
        "schema_version": "N72R13_WINDOW_EXPORT_MANIFEST_V1",
        "status": "PASS_EXPORT_N72R13",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": _sha(protocol_path),
        "source_frozen_e1b_manifest": str(r5_manifest_path),
        "source_frozen_e1b_manifest_sha256": _sha(r5_manifest_path),
        "source_oracle_root": str(oracle_root),
        "source_oracle_metrics": str(oracle_root / "temporal_oracle_metrics.json"),
        "trackeval_root": str(trackeval_root),
        "trackeval_commit": trackeval_commit,
        "source_events": 32,
        "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
        "horizons": list(HORIZONS),
        "logical_variants": list(VARIANTS),
        "pseudo_sequence_count": len(windows),
        "record_count": len(records),
        "event_windows": windows,
        "records": records,
        "runtime_sources": source_records,
        "integrity": {
            "expected_records": expected,
            "actual_records": len(records),
            "duplicate_keys": duplicates,
            "missing_keys": missing,
            "all_runtime_axes_verified": True,
            "all_runtime_gt_flags_false": True,
            "all_prediction_files_written_atomically": True,
        },
        "runtime_future_gt_used": False,
        "gt_used_only_for_posthoc_evaluation": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "oracle_semantics": "sequential_TIV_oracle_allows_ONLY_posthoc_KEEP_or_APPLY_decisions; compare separately from frozen E0/E1B",
        "oracle_effective_apply_count": 0,
        "oracle_effective_opportunity_count": 176,
    }
    _atomic_json(output_root / "export_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R13/trackeval")
    parser.add_argument("--data-root", type=Path, default=Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R9/protocol.json")
    parser.add_argument("--frozen-e1b-manifest", type=Path, default=ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json")
    parser.add_argument("--oracle-root", type=Path, default=ROOT / "outputs/N72R13/temporal_oracle")
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    try:
        manifest = export(output_root, args.data_root.resolve(), args.protocol.resolve(), args.frozen_e1b_manifest.resolve(), args.oracle_root.resolve(), args.trackeval_root.resolve())
        print(json.dumps({"status": manifest["status"], "records": manifest["record_count"], "output": str(output_root / "export_manifest.json")}, sort_keys=True))
        return 0
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (output_root / f"export_failure_attempt{attempt}.json").exists():
            attempt += 1
        _atomic_json(output_root / f"export_failure_attempt{attempt}.json", {
            "schema_version": "N72R13_EXPORT_FAILURE_V1",
            "status": "FAIL_EXPORT_N72R13",
            "command": [sys.executable, *sys.argv],
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_outputs_modified": False,
        })
        print(json.dumps({"status": "FAIL_EXPORT_N72R13", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
