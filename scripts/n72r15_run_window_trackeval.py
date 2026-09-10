#!/usr/bin/env python3
"""Run pinned official TrackEval on sealed N72R15 interaction windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r14_run_window_trackeval as legacy_run


HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1I_HUMAN_RELATIVE_STATE",
    "E1J_TRUSTED_GLOBAL_RELATIVE_STATE",
)
PINNED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
DEFAULT_ROOT = ROOT / "outputs/N72R15/trackeval_attempt_01"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise legacy_run.WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    legacy_run.write_json_atomic(path, value)


def _validate_export(export: Path, trackeval: Path) -> dict[str, Any]:
    manifest = _read(export / "export_manifest.json")
    if manifest.get("status") != "PASS_EXPORT_N72R15":
        raise legacy_run.WindowTrackEvalError(f"N72R15 export is not PASS: {manifest.get('status')}")
    if int(manifest.get("source_events", -1)) != 32 or int(manifest.get("record_count", -1)) != 32 * 3 * 3:
        raise legacy_run.WindowTrackEvalError("N72R15 export does not contain 32 events and 288 records")
    if manifest.get("runtime_future_gt_used") is not False:
        raise legacy_run.WindowTrackEvalError("export manifest has runtime future GT")
    commit = __import__("subprocess").check_output(["git", "-C", str(trackeval), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_TRACKEVAL_COMMIT:
        raise legacy_run.WindowTrackEvalError(f"TrackEval commit {commit} != pinned {PINNED_TRACKEVAL_COMMIT}")
    return manifest


def run(export: Path, output: Path, trackeval: Path) -> dict[str, Any]:
    legacy_run.VARIANTS = VARIANTS
    legacy_run.PINNED_TRACKEVAL_COMMIT = PINNED_TRACKEVAL_COMMIT
    export_manifest = _validate_export(export, trackeval)
    results = [legacy_run.run_horizon(export, output, trackeval, horizon) for horizon in HORIZONS]
    rows = [row for result in results if result.get("status") == "PASS" for row in result.get("records", [])]
    expected = 32 * len(HORIZONS) * len(VARIANTS)
    keys = [(row["event_id"], row["logical_variant"], int(row["horizon"])) for row in rows]
    expected_keys = {
        (str(window["event_id"]), variant, int(window["horizon"]))
        for window in export_manifest.get("event_windows", [])
        for variant in VARIANTS
    }
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    missing_keys = sorted(expected_keys - set(keys))
    failed = [result for result in results if result.get("status") != "PASS"]
    status = "PASS_TRACKEVAL_N72R15" if not failed and len(rows) == expected and not duplicate_keys and not missing_keys else "BLOCKED_INCOMPLETE_TRACKEVAL_N72R15"
    manifest = {
        "schema_version": "N72R15_TRACKEVAL_RUN_MANIFEST_V1",
        "status": status,
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "horizons": list(HORIZONS),
        "logical_variants": list(VARIANTS),
        "expected_record_count": expected,
        "record_count": len(rows),
        "duplicate_keys": duplicate_keys,
        "missing_keys": missing_keys,
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
    _write_json(output / "trackeval_run_manifest.json", manifest)
    legacy_run.write_text_atomic(
        output / ("per_event_metrics.jsonl" if status == "PASS_TRACKEVAL_N72R15" else "per_event_metrics_partial.jsonl"),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    export = args.export_root.resolve()
    output = args.output_root.resolve()
    trackeval = args.trackeval_root.resolve()
    try:
        if tuple(sorted(args.horizons)) != HORIZONS:
            raise legacy_run.WindowTrackEvalError(f"must run exactly H20/H50/H100: {args.horizons}")
        manifest = run(export, output, trackeval)
        stage = {
            "schema_version": "N72R15_STAGE_STATUS_V1",
            "stage": "N72R15-10-OFFICIAL-TRACKEVAL",
            "status": manifest["status"],
            "manifest": str(output / "trackeval_run_manifest.json"),
            "record_count": int(manifest["record_count"]),
            "expected_record_count": int(manifest["expected_record_count"]),
            "duplicate_key_count": len(manifest["duplicate_keys"]),
            "missing_key_count": len(manifest["missing_keys"]),
            "horizon_statuses": {str(result["horizon"]): result["status"] for result in manifest["horizon_results"]},
            "trackeval_commit": PINNED_TRACKEVAL_COMMIT,
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        _write_json(output.parent / "stage_10_status.json", stage)
        print(json.dumps(stage, sort_keys=True))
        return 0 if manifest["status"] == "PASS_TRACKEVAL_N72R15" else 1
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_TRACKEVAL_RUNNER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        }
        _write_json(output / "trackeval_failure.json", failure)
        _write_json(output.parent / "stage_10_status.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
