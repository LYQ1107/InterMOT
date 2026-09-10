#!/usr/bin/env python3
"""Export sealed N72R14 windows for the pinned official TrackEval."""

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
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation import window_trackeval as wt  # noqa: E402


HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1F_TARGET_POOL_ONLY",
    "E1G_TARGET_STATE_ONLY",
    "E1H_PERSISTENT_GLOBAL_STATE",
)
PINNED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
DEFAULT_MANIFEST = ROOT / "outputs/N72R14/formal_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R14/trackeval"
DEFAULT_DATA = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _runtime_scan(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested is not False:
                errors.append(f"{location}/{key}={nested!r}")
            errors.extend(_runtime_scan(nested, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_runtime_scan(nested, f"{location}/{index}"))
    return errors


def _load_rows(record: Mapping[str, Any], event_id: str, variant: str) -> list[dict[str, Any]]:
    path = Path(str(record["frames"]))
    if not path.is_file():
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: runtime frames are missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 101:
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: expected 101 rows, got {len(rows)}")
    event_frame = int(rows[0].get("event_frame", -1))
    if [int(row.get("frame", -1)) for row in rows] != list(range(event_frame, event_frame + 101)):
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: frame axis is incomplete")
    if _runtime_scan(rows):
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: runtime GT flags are present")
    if rows[0].get("record_kind") != "event_frame_correction" or rows[0].get("memory_read") is not False:
        raise wt.WindowTrackEvalError(f"{event_id}/{variant}: event-frame causal boundary failed")
    return rows


def export(*, output_root: Path, formal_manifest: Path, data_root: Path, trackeval_root: Path) -> dict[str, Any]:
    formal = read_json(formal_manifest)
    if formal.get("status") != "PASS_N72R14_FORMAL_REPLAY" or int(formal.get("event_count", -1)) != 32:
        raise wt.WindowTrackEvalError("N72R14 formal manifest is not complete")
    try:
        commit = subprocess.check_output(["git", "-C", str(trackeval_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise wt.WindowTrackEvalError(f"cannot resolve TrackEval commit: {exc}") from exc
    if commit != PINNED_TRACKEVAL_COMMIT:
        raise wt.WindowTrackEvalError(f"TrackEval commit {commit} != pinned {PINNED_TRACKEVAL_COMMIT}")

    output_root.mkdir(parents=True, exist_ok=True)
    events: dict[str, dict[str, Any]] = {}
    rows_by_event: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_records: list[dict[str, Any]] = []
    for event_record in formal.get("events", []):
        event_id = str(event_record["event_id"])
        if event_id in events:
            raise wt.WindowTrackEvalError(f"duplicate event {event_id}")
        if event_record.get("status") != "PASS_N72R14_FORMAL_EVENT":
            raise wt.WindowTrackEvalError(f"{event_id}: event is not PASS")
        events[event_id] = dict(event_record)
        rows_by_event[event_id] = {}
        for variant_record in event_record.get("variants", []):
            variant = str(variant_record["variant"])
            if variant not in VARIANTS or variant in rows_by_event[event_id]:
                raise wt.WindowTrackEvalError(f"{event_id}: invalid or duplicate variant {variant}")
            rows = _load_rows(variant_record, event_id, variant)
            rows_by_event[event_id][variant] = rows
            path = Path(str(variant_record["frames"]))
            source_records.append({"event_id": event_id, "variant": variant, "path": str(path), "sha256": sha256_file(path)})
        if set(rows_by_event[event_id]) != set(VARIANTS):
            raise wt.WindowTrackEvalError(f"{event_id}: variant set incomplete")
    if len(events) != 32 or len({str(item["sequence"]) for item in events.values()}) != 18:
        raise wt.WindowTrackEvalError("formal event/sequence counts are not 32/18")

    windows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seqmaps: dict[int, list[str]] = {horizon: [] for horizon in HORIZONS}
    for event_id in sorted(events):
        event = events[event_id]
        event_frame = int(event["event_frame"])
        source_event = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": event_frame,
            "split": "train",
            "frozen_event": event,
        }
        protocol_payload = read_json(ROOT / "outputs/N72R9/protocol.json")
        frozen_event = next(
            item for item in protocol_payload.get("source_event_selection", {}).get("events", [])
            if str(item["event_id"]) == event_id
        )
        source_event.update(frozen_event)
        source_event["frozen_event"] = frozen_event
        for horizon in HORIZONS:
            pseudo = wt._pseudo_name(event_id, horizon)
            resolved = wt.resolve_source_gt_and_seqinfo(data_root, source_event, project_root=ROOT)
            gt_source = Path(resolved["gt_path"])
            seqinfo_source = Path(resolved["seqinfo_path"])
            gt_dir = output_root / "pseudo_gt" / pseudo
            wt.write_text_atomic(
                gt_dir / "gt" / "gt.txt",
                "\n".join(wt._gt_window_lines(gt_source, event_frame=event_frame, horizon=horizon, event_id=event_id)) + "\n",
            )
            wt.write_text_atomic(gt_dir / "seqinfo.ini", wt._seqinfo_text(seqinfo_source, pseudo, horizon))
            seqmaps[horizon].append(pseudo)
            windows.append({
                "pseudo_sequence": pseudo,
                "original_sequence": str(event["sequence"]),
                "event_id": event_id,
                "action_type": str(event["action_type"]),
                "event_frame": event_frame,
                "horizon": horizon,
                "original_start_frame": event_frame + 1,
                "original_end_frame": event_frame + horizon,
                "gt_path": str(gt_dir / "gt" / "gt.txt"),
                "seqinfo_path": str(gt_dir / "seqinfo.ini"),
                "gt_source_path": str(gt_source),
                "seqinfo_source_path": str(seqinfo_source),
                "gt_source_resolution": resolved["resolution"],
                "gt_sha256": sha256_file(gt_dir / "gt" / "gt.txt"),
            })
            for variant in VARIANTS:
                prediction_lines, counts = wt._prediction_lines(
                    rows_by_event[event_id][variant],
                    event_frame=event_frame,
                    horizon=horizon,
                    event_id=event_id,
                    variant=variant,
                )
                prediction_path = output_root / "trackers" / variant / "data" / f"{pseudo}.txt"
                wt.write_text_atomic(prediction_path, "\n".join(prediction_lines) + ("\n" if prediction_lines else ""))
                source = next(item for item in source_records if item["event_id"] == event_id and item["variant"] == variant)
                records.append({
                    "logical_variant": variant,
                    "runtime_frames_path": source["path"],
                    "runtime_frames_sha256": source["sha256"],
                    "pseudo_sequence": pseudo,
                    "original_sequence": str(event["sequence"]),
                    "event_id": event_id,
                    "action_type": str(event["action_type"]),
                    "event_frame": event_frame,
                    "horizon": horizon,
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    **counts,
                })
    for horizon, names in seqmaps.items():
        wt.write_text_atomic(output_root / "seqmaps" / f"seqmap_h{horizon:03d}.txt", "name\n" + "\n".join(names) + "\n")
    expected = 32 * len(HORIZONS) * len(VARIANTS)
    keys = [(item["event_id"], item["logical_variant"], int(item["horizon"])) for item in records]
    expected_keys = {(event_id, variant, horizon) for event_id in events for variant in VARIANTS for horizon in HORIZONS}
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    missing = sorted(expected_keys - set(keys))
    if len(records) != expected or duplicates or missing:
        raise wt.WindowTrackEvalError(f"export completeness failed: records={len(records)} duplicates={duplicates} missing={missing[:5]}")
    manifest = {
        "schema_version": "N72R14_WINDOW_TRACKEVAL_EXPORT_V1",
        "status": "PASS_EXPORT_N72R14",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "source_formal_manifest": str(formal_manifest),
        "source_formal_manifest_sha256": sha256_file(formal_manifest),
        "source_protocol": str(ROOT / "outputs/N72R9/protocol.json"),
        "source_protocol_sha256": sha256_file(ROOT / "outputs/N72R9/protocol.json"),
        "trackeval_root": str(trackeval_root),
        "trackeval_commit": commit,
        "data_root": str(data_root),
        "source_events": len(events),
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
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "export_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    args = parser.parse_args()
    # Keep stage status beside the attempt root.  This lets a corrected
    # attempt run under outputs/N72R14/attempt_02 without overwriting the
    # first export failure recorded at outputs/N72R14/stage_05_status.json.
    stage_status_path = args.output_root.resolve().parent / "stage_05_status.json"
    try:
        manifest = export(
            output_root=args.output_root.resolve(),
            formal_manifest=args.formal_manifest.resolve(),
            data_root=args.data_root.resolve(),
            trackeval_root=args.trackeval_root.resolve(),
        )
        status = {
            "schema_version": "N72R14_STAGE_STATUS_V1",
            "stage": "N72R14-05-TRACKEVAL-EXPORT",
            "status": manifest["status"],
            "manifest": str(args.output_root.resolve() / "export_manifest.json"),
            "record_count": manifest["record_count"],
            "expected_record_count": 32 * len(HORIZONS) * len(VARIANTS),
            "trackeval_commit": manifest["trackeval_commit"],
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(stage_status_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R14_FAILURE_V1",
            "status": "FAIL_N72R14_TRACKEVAL_EXPORT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(args.output_root.resolve() / "export_failure.json", failure)
        atomic_json(stage_status_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
