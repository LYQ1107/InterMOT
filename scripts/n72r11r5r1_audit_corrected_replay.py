#!/usr/bin/env python3
"""Strictly audit N72R11R5R1 corrected replay artifacts.

The audit is runtime-only for the frame/solver checks: it reads frozen source
metadata and sealed runtime JSONL, but never opens dataset GT.  Posthoc files
are checked only for their sealed provenance/status so the existing frozen
metric aggregator remains the sole implementation of causal scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r11_on_demand_replay as on_demand  # noqa: E402
from scripts import n72r9_temporal_replay as legacy  # noqa: E402


TREATMENTS = {
    "e1a": "E1A_EXACT_ONPOLICY_V3_LEGACY",
    "e1b": "E1B_PCTIS_LEGACY",
}
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_E1A = ROOT / "outputs/N72R11R5R1/formal_e1a_manifest.json"
DEFAULT_E1B = ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R5R1/corrected_replay_audit.json"
DEFAULT_STAGE = ROOT / "outputs/N72R11R5R1/stage_03_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _runtime_forbidden_scan(value: Any, location: str, errors: list[str]) -> None:
    forbidden = {
        "dataset_gt_id",
        "gt_box",
        "future_identity_error",
        "future_iou",
        "future_gt",
        "gt_target",
        "gt_id",
    }
    flags = {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in forbidden:
                errors.append(f"{location}/{key}")
            if key_text in flags and nested is not False:
                errors.append(f"{location}/{key}=not_false")
            _runtime_forbidden_scan(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _runtime_forbidden_scan(nested, f"{location}/{index}", errors)


def _positive_box(value: Any) -> bool:
    try:
        box = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(box.size == 4 and np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1])


def _audit_manifest(kind: str, manifest_path: Path, expected_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    treatment = TREATMENTS[kind]
    manifest = read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    records = manifest.get("records")
    expected_by_id = {str(item["event_id"]): item for item in expected_events}
    if manifest.get("status") != "PASS_ALL_SELECTED":
        failures.append({"scope": "manifest", "error": f"status={manifest.get('status')!r}"})
    if int(manifest.get("event_count", -1)) != len(expected_events):
        failures.append({"scope": "manifest", "error": "event_count_mismatch", "value": manifest.get("event_count")})
    if manifest.get("require_positive_geometry") is not True or manifest.get("baseline_regenerated_from_frozen_sources") is not True:
        failures.append({"scope": "manifest", "error": "geometry_replay_flags_missing"})
    if not isinstance(records, list):
        failures.append({"scope": "manifest", "error": "records_missing"})
        records = []
    seen: set[str] = set()
    event_results: list[dict[str, Any]] = []
    total_filtered = 0
    filtered_by_source: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            failures.append({"scope": "manifest", "error": "non_object_record"})
            continue
        event_id = str(record.get("event_id"))
        if event_id in seen:
            failures.append({"event_id": event_id, "error": "duplicate_event_id"})
            continue
        seen.add(event_id)
        event_failure_count_before = len(failures)
        if event_id not in expected_by_id:
            failures.append({"event_id": event_id, "error": "unexpected_event_id"})
            continue
        expected = expected_by_id[event_id]
        if record.get("status") != "PASS":
            failures.append({"event_id": event_id, "error": f"manifest_record_status={record.get('status')!r}"})
            continue
        done_path = resolve(record.get("done"))
        if not done_path.is_file():
            failures.append({"event_id": event_id, "error": "done_missing", "path": str(done_path)})
            continue
        if str(record.get("done_sha256")) != sha256_file(done_path):
            failures.append({"event_id": event_id, "error": "done_hash_mismatch"})
            continue
        done = read_json(done_path)
        if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
            failures.append({"event_id": event_id, "error": f"done_status={done.get('status')!r}"})
        if done.get("runtime_future_gt_used") is not False or done.get("posthoc_gt_used") is not True:
            failures.append({"event_id": event_id, "error": "done_provenance"})
        if done.get("require_positive_geometry") is not True or done.get("baseline_regenerated_from_frozen_sources") is not True:
            failures.append({"event_id": event_id, "error": "done_geometry_flags"})
        if str(done.get("scorer_kind")) != str(manifest.get("scorer_kind")):
            failures.append({"event_id": event_id, "error": "done_scorer_kind_mismatch"})
        if resolve(done.get("model_checkpoint")) != resolve(manifest.get("model_checkpoint")):
            failures.append({"event_id": event_id, "error": "done_checkpoint_mismatch"})
        posthoc_path = resolve(done.get("posthoc"))
        if not posthoc_path.is_file():
            failures.append({"event_id": event_id, "error": "posthoc_missing"})
        else:
            posthoc = read_json(posthoc_path)
            if posthoc.get("status") != "PASS_N72R11_POSTHOC_EVENT" or posthoc.get("posthoc_gt_used") is not True or posthoc.get("runtime_future_gt_used") is not False:
                failures.append({"event_id": event_id, "error": "posthoc_provenance_or_status"})
        inputs = on_demand._load_inputs(expected, horizon=100)
        inputs = dict(inputs)
        inputs["require_positive_geometry"] = True
        inputs["baseline_regenerated_from_frozen_sources"] = True
        runtime_seal_path = resolve(done.get("runtime_event_sealed"))
        seal = read_json(runtime_seal_path)
        if seal.get("runtime_future_gt_used") is not False or seal.get("runtime_gt_read") is not False or seal.get("gt_loaded") is not False:
            failures.append({"event_id": event_id, "error": "runtime_seal_provenance"})
        variant_audit: dict[str, Any] = {}
        expected_variants = ("E0_BASELINE_B0", treatment)
        for variant in expected_variants:
            variant_manifest = seal.get("runtime_manifests", {}).get(variant)
            if not isinstance(variant_manifest, Mapping):
                failures.append({"event_id": event_id, "variant": variant, "error": "runtime_manifest_missing"})
                continue
            frames_path = resolve(variant_manifest.get("frames"))
            if not frames_path.is_file():
                failures.append({"event_id": event_id, "variant": variant, "error": "runtime_frames_missing"})
                continue
            if str(variant_manifest.get("frames_sha256")) != sha256_file(frames_path):
                failures.append({"event_id": event_id, "variant": variant, "error": "runtime_frames_hash_mismatch"})
                continue
            rows = read_jsonl(frames_path)
            try:
                if variant == "E0_BASELINE_B0":
                    legacy._validate_runtime_rows(rows, inputs, variant)
                else:
                    on_demand._validate_runtime(rows, inputs, variant)
            except Exception as exc:
                failures.append({"event_id": event_id, "variant": variant, "error": "runtime_validator", "detail": str(exc)})
            runtime_errors: list[str] = []
            filtered = 0
            source_counts: dict[str, int] = {}
            positive_output_rows = 0
            for row in rows:
                _runtime_forbidden_scan(row, f"{variant}/{row.get('frame')}", runtime_errors)
                pool = row.get("candidate_pool")
                if not isinstance(pool, Mapping):
                    continue
                filtered += int(pool.get("invalid_geometry_candidate_count", 0))
                for rejected in pool.get("invalid_geometry_candidates", []):
                    if isinstance(rejected, Mapping):
                        source = str(rejected.get("candidate_source"))
                        source_counts[source] = source_counts.get(source, 0) + 1
                for output_row in row.get("candidate_rows", []):
                    if _positive_box(output_row.get("box_xyxy")):
                        positive_output_rows += 1
            if runtime_errors:
                failures.append({"event_id": event_id, "variant": variant, "error": "runtime_forbidden_fields", "fields": sorted(set(runtime_errors))[:16]})
            stats = done.get("baseline_geometry_stats") if variant == "E0_BASELINE_B0" else done.get("stats", {}).get(variant)
            if not isinstance(stats, Mapping) or stats.get("require_positive_geometry") is not True:
                failures.append({"event_id": event_id, "variant": variant, "error": "variant_geometry_stats_missing"})
            elif int(stats.get("geometry_filtered_candidate_count", -1)) != filtered:
                failures.append({"event_id": event_id, "variant": variant, "error": "filtered_count_mismatch", "rows": filtered, "stats": stats.get("geometry_filtered_candidate_count")})
            total_filtered += filtered
            for source, count in source_counts.items():
                filtered_by_source[source] = filtered_by_source.get(source, 0) + count
            variant_audit[variant] = {
                "frames": len(rows),
                "runtime_frames_sha256": sha256_file(frames_path),
                "filtered_geometry_candidates": filtered,
                "filtered_geometry_by_source": source_counts,
                "positive_output_row_count": positive_output_rows,
                "runtime_future_gt_used": False,
                "validator_error_count": len(runtime_errors),
            }
        event_results.append(
            {
                "event_id": event_id,
                "sequence": str(record.get("sequence")),
                "action_type": str(record.get("action_type")),
                "new_failures": len(failures) - event_failure_count_before,
                "variants": variant_audit,
            }
        )
    missing = sorted(set(expected_by_id) - seen)
    for event_id in missing:
        failures.append({"event_id": event_id, "error": "missing_manifest_record"})
    return {
        "kind": kind,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": manifest.get("status"),
        "record_count": len(records),
        "unique_record_count": len(seen),
        "missing_event_count": len(missing),
        "duplicate_event_count": len(records) - len(seen),
        "event_results": event_results,
        "total_filtered_geometry_candidates": total_filtered,
        "filtered_geometry_by_source": filtered_by_source,
        "failures": failures,
        "pass": not failures and seen == set(expected_by_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1a-manifest", type=Path, default=DEFAULT_E1A)
    parser.add_argument("--e1b-manifest", type=Path, default=DEFAULT_E1B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-status", type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    e1a_manifest = args.e1a_manifest if args.e1a_manifest.is_absolute() else ROOT / args.e1a_manifest
    e1b_manifest = args.e1b_manifest if args.e1b_manifest.is_absolute() else ROOT / args.e1b_manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    stage_status = args.stage_status if args.stage_status.is_absolute() else ROOT / args.stage_status
    events = [dict(item) for item in read_json(PROTOCOL).get("source_event_selection", {}).get("events", [])]
    results = {
        kind: _audit_manifest(kind, path, events)
        for kind, path in (("e1a", e1a_manifest), ("e1b", e1b_manifest))
    }
    result = {
        "schema_version": "N72R11R5R1_CORRECTED_REPLAY_AUDIT_V1",
        "status": "PASS_CORRECTED_REPLAY_RUNTIME_AUDIT" if all(item["pass"] for item in results.values()) else "FAIL_CORRECTED_REPLAY_RUNTIME_AUDIT",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "event_count": len(events),
        "runtime_future_gt_used": False,
        "results": results,
    }
    atomic_json(output, result)
    atomic_json(
        stage_status,
        {
            "schema_version": "N72R11R5R1_STAGE_03_STATUS_V1",
            "stage": "N72R11R5R1_STAGE_03_CORRECTED_REPLAY_AUDIT",
            "status": result["status"],
            "created_at_utc": now_utc(),
            "audit": str(output),
            "audit_sha256": sha256_file(output),
            "event_count": len(events),
            "e1a": {key: results["e1a"][key] for key in ("record_count", "unique_record_count", "missing_event_count", "duplicate_event_count", "total_filtered_geometry_candidates", "pass")},
            "e1b": {key: results["e1b"][key] for key in ("record_count", "unique_record_count", "missing_event_count", "duplicate_event_count", "total_filtered_geometry_candidates", "pass")},
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        },
    )
    print(json.dumps({"status": result["status"], "output": str(output), "e1a_pass": results["e1a"]["pass"], "e1b_pass": results["e1b"]["pass"]}, sort_keys=True))
    return 0 if result["status"] == "PASS_CORRECTED_REPLAY_RUNTIME_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
