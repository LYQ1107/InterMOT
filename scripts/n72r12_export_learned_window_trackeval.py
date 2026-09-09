#!/usr/bin/env python3
"""Export the fixed E0/E1B/E1C/E1D matrix for official TrackEval."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation import window_trackeval as wt  # noqa: E402
from scripts.n72r12_export_window_trackeval import (  # noqa: E402
    _atomic_json,
    _read,
    _runtime_entry,
    _sha,
)

HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1B_PCTIS",
    "E1C_PCTIS_SAFE",
    "E1D_PCTIS_LEARNED_SAFE",
)
EXPECTED_EVENTS = 32


def export(
    output_root: Path,
    data_root: Path,
    r5_manifest_path: Path,
    safe_manifest_path: Path,
    learned_manifest_path: Path,
    trackeval_root: Path,
) -> dict[str, Any]:
    r5 = _read(r5_manifest_path)
    safe = _read(safe_manifest_path)
    learned = _read(learned_manifest_path)
    if r5.get("status") != "PASS_ALL_SELECTED" or r5.get("event_count") != EXPECTED_EVENTS:
        raise wt.WindowTrackEvalError("R5R1 E1B source manifest is incomplete")
    if safe.get("status") != "PASS_N72R12_FORMAL_RUNTIME" or safe.get("event_count") != EXPECTED_EVENTS:
        raise wt.WindowTrackEvalError("N72R12 E1C source manifest is incomplete")
    if learned.get("status") != "PASS_N72R12_LEARNED_RUNTIME" or learned.get("event_count") != EXPECTED_EVENTS:
        raise wt.WindowTrackEvalError("N72R12 E1D source manifest is incomplete")
    r5_records = {str(item["event_id"]): item for item in r5["records"]}
    safe_records = {str(item["event_id"]): item for item in safe["records"]}
    learned_records = {str(item["event_id"]): item for item in learned["records"]}
    protocol_path = ROOT / "outputs/N72R9/protocol.json"
    protocol = _read(protocol_path)
    events = {str(item["event_id"]): dict(item) for item in protocol["source_event_selection"]["events"]}
    if (
        len(events) != EXPECTED_EVENTS
        or set(events) != set(r5_records)
        or set(events) != set(safe_records)
        or set(events) != set(learned_records)
    ):
        raise wt.WindowTrackEvalError("learned export event key sets are not exactly the frozen 32")
    output_root.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seqmaps: dict[int, list[str]] = {horizon: [] for horizon in HORIZONS}
    for event_id in sorted(events):
        event = events[event_id]
        old_dir = Path(str(r5_records[event_id]["done"])).parent
        safe_dir = Path(str(safe_records[event_id]["done"])).parent
        learned_dir = Path(str(learned_records[event_id]["e1d_done"])).parent
        entries = {
            "E0_BASELINE_B0": _runtime_entry(old_dir / "done.json", "E0_BASELINE_B0")[0],
            "E1B_PCTIS": _runtime_entry(old_dir / "done.json", "E1B_PCTIS_LEGACY")[0],
            "E1C_PCTIS_SAFE": _runtime_entry(safe_dir / "done.json", "E1C_PCTIS_SAFE")[0],
            "E1D_PCTIS_LEARNED_SAFE": _runtime_entry(learned_dir / "done.json", "E1D_PCTIS_LEARNED_SAFE")[0],
        }
        runtime_rows = {
            logical: wt._load_runtime_rows(entries[logical], event_id, entries[logical].get("variant", logical))
            for logical in VARIANTS
        }
        for horizon in HORIZONS:
            pseudo = wt._pseudo_name(event_id, horizon)
            resolved = wt.resolve_source_gt_and_seqinfo(data_root, {**event, "frozen_event": event}, project_root=ROOT)
            gt_source = Path(resolved["gt_path"])
            seqinfo_source = Path(resolved["seqinfo_path"])
            gt_lines = wt._gt_window_lines(gt_source, event_frame=int(event["event_frame"]), horizon=horizon, event_id=event_id)
            gt_dir = output_root / "pseudo_gt" / pseudo
            wt.write_text_atomic(gt_dir / "gt" / "gt.txt", "\n".join(gt_lines) + "\n")
            wt.write_text_atomic(gt_dir / "seqinfo.ini", wt._seqinfo_text(seqinfo_source, pseudo, horizon))
            seqmaps[horizon].append(pseudo)
            windows.append({
                "pseudo_sequence": pseudo,
                "original_sequence": event["sequence"],
                "event_id": event_id,
                "action_type": event["action_type"],
                "event_frame": int(event["event_frame"]),
                "horizon": horizon,
                "original_start_frame": int(event["event_frame"]) + 1,
                "original_end_frame": int(event["event_frame"]) + horizon,
                "gt_path": str(gt_dir / "gt" / "gt.txt"),
                "seqinfo_path": str(gt_dir / "seqinfo.ini"),
                "gt_source_path": str(gt_source),
                "seqinfo_source_path": str(seqinfo_source),
                "gt_source_resolution": resolved["resolution"],
                "gt_sha256": _sha(gt_dir / "gt" / "gt.txt"),
            })
            for logical in VARIANTS:
                lines, counts = wt._prediction_lines(runtime_rows[logical], event_frame=int(event["event_frame"]), horizon=horizon, event_id=event_id, variant=logical)
                prediction_path = output_root / "trackers" / logical / "data" / f"{pseudo}.txt"
                wt.write_text_atomic(prediction_path, "\n".join(lines) + ("\n" if lines else ""))
                records.append({
                    "logical_variant": logical,
                    "sealed_runtime_variant": entries[logical].get("variant", logical),
                    "runtime_frames_path": entries[logical]["frames"],
                    "runtime_frames_sha256": entries[logical]["frames_sha256"],
                    "pseudo_sequence": pseudo,
                    "original_sequence": event["sequence"],
                    "event_id": event_id,
                    "action_type": event["action_type"],
                    "event_frame": int(event["event_frame"]),
                    "horizon": horizon,
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": _sha(prediction_path),
                    **counts,
                })
    for horizon, names in seqmaps.items():
        wt.write_text_atomic(output_root / "seqmaps" / f"seqmap_h{horizon:03d}.txt", "name\n" + "\n".join(names) + "\n")
    expected = EXPECTED_EVENTS * len(VARIANTS) * len(HORIZONS)
    keys = [(item["event_id"], item["logical_variant"], int(item["horizon"])) for item in records]
    expected_keys = {(event_id, logical, horizon) for event_id in events for logical in VARIANTS for horizon in HORIZONS}
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    missing = sorted(expected_keys - set(keys))
    if len(records) != expected or duplicates or missing:
        raise wt.WindowTrackEvalError(f"N72R12 learned export completeness failed records={len(records)} duplicates={duplicates} missing={missing[:5]}")
    try:
        trackeval_commit = subprocess.check_output(["git", "-C", str(trackeval_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        trackeval_commit = None
    manifest = {
        "schema_version": "N72R12_LEARNED_WINDOW_EXPORT_MANIFEST_V1",
        "status": "PASS_EXPORT_N72R12_LEARNED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": _sha(protocol_path),
        "source_r5r1_manifest": str(r5_manifest_path),
        "source_r5r1_manifest_sha256": _sha(r5_manifest_path),
        "source_safe_manifest": str(safe_manifest_path),
        "source_safe_manifest_sha256": _sha(safe_manifest_path),
        "source_learned_manifest": str(learned_manifest_path),
        "source_learned_manifest_sha256": _sha(learned_manifest_path),
        "trackeval_root": str(trackeval_root),
        "trackeval_commit": trackeval_commit,
        "source_events": EXPECTED_EVENTS,
        "independent_sequence_count": len({str(event["sequence"]) for event in events.values()}),
        "horizons": list(HORIZONS),
        "logical_variants": list(VARIANTS),
        "pseudo_sequence_count": len(windows),
        "record_count": len(records),
        "event_windows": windows,
        "records": records,
        "integrity": {
            "expected_records": expected,
            "actual_records": len(records),
            "duplicate_keys": duplicates,
            "missing_keys": missing,
            "all_runtime_hashes_verified": True,
            "all_prediction_files_written_atomically": True,
        },
        "runtime_future_gt_used": False,
        "gt_used_only_for_posthoc_evaluation": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    _atomic_json(output_root / "export_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R12/trackeval_learned")
    parser.add_argument("--data-root", type=Path, default=Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"))
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    args = parser.parse_args()
    try:
        manifest = export(
            args.output_root.resolve(),
            args.data_root.resolve(),
            ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json",
            ROOT / "outputs/N72R12/formal_safe/formal_safe_manifest.json",
            ROOT / "outputs/N72R12/formal_learned/formal_learned_manifest.json",
            args.trackeval_root.resolve(),
        )
        print(json.dumps({"status": manifest["status"], "records": manifest["record_count"], "output": str(args.output_root.resolve() / "export_manifest.json")}, sort_keys=True))
        return 0
    except Exception as exc:
        root = args.output_root.resolve()
        attempt = 1
        while (root / f"export_failure_attempt{attempt}.json").exists():
            attempt += 1
        _atomic_json(root / f"export_failure_attempt{attempt}.json", {"schema_version": "N72R12_LEARNED_EXPORT_FAILURE_V1", "status": "FAIL_EXPORT_N72R12_LEARNED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "historical_outputs_modified": False})
        print(json.dumps({"status": "FAIL_EXPORT_N72R12_LEARNED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
