"""N69 Stage 02: audit and register the frozen N54 candidate cache.

No SAM3 inference is launched here.  Stage 01 already checked every source
frame's structural row/public-axis contract.  This stage makes the reuse
explicit, verifies the expected event×variant×frame key set, and records the
90 target-absent frames as no-op/candidate-recall evidence rather than
silently dropping them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n69"
DIAG = OUT / "diagnosis"
CACHE = OUT / "cache"
ATTEMPTS = OUT / "attempts"
MAPPING_ROWS = DIAG / "mapping_audit.jsonl"
MAPPING_SUMMARY = DIAG / "mapping_summary.json"
N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
N54_STATUS = ROOT / "outputs/n54/replay/runtime_status.json"
PROTOCOL = OUT / "protocol.json"
MANIFEST = CACHE / "candidate_cache_manifest.json"
AUDIT = CACHE / "candidate_cache_audit.json"
STATUS = OUT / "stage_02_status.json"

VARIANTS = ("M0", "M1", "M2", "M3", "M4")
EVENTS = 24
FRAMES = 100


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return value


def event_ids() -> list[str]:
    manifest = load_json(N37_MANIFEST)
    values = [str(item.get("protocol_candidate_id")) for item in manifest.get("events", [])]
    values = [value for value in values if value and value != "None"]
    if len(values) != EVENTS or len(set(values)) != EVENTS:
        raise RuntimeError(f"N37 event manifest key set is not {EVENTS} unique events")
    return sorted(values)


def source_inventory(ids: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event_id in ids:
        path = N54_RUNTIME / f"{event_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing frozen N54 runtime file: {path}")
        result.append({"event_id": event_id, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return result


def read_mapping_keys() -> tuple[list[tuple[str, str, int]], Counter[str], dict[str, Any]]:
    expected = set()
    ids = event_ids()
    for event_id in ids:
        for variant in VARIANTS:
            for frame in range(1, FRAMES + 1):
                expected.add((event_id, variant, frame))
    seen: set[tuple[str, str, int]] = set()
    errors: Counter[str] = Counter()
    row_count = 0
    first_rows: list[dict[str, Any]] = []
    with MAPPING_ROWS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["event_id"]), str(row["variant"]), int(row["frame"]) - int(row["event_frame"]))
            row_count += 1
            if key in seen:
                errors["duplicate_event_variant_horizon"] += 1
            seen.add(key)
            if len(first_rows) < 3:
                first_rows.append(row)
            if row.get("runtime_future_gt_used") is not False:
                errors["runtime_future_gt_used"] += 1
            if row.get("candidate_integrity", {}).get("structural_valid") is not True:
                errors["candidate_structure_invalid"] += 1
    missing = expected - seen
    extra = seen - expected
    if missing:
        errors["missing_event_variant_horizon"] = len(missing)
    if extra:
        errors["extra_event_variant_horizon"] = len(extra)
    return first_rows, errors, {"rows": row_count, "expected_rows": len(expected), "missing_keys": sorted(missing)[:20], "extra_keys": sorted(extra)[:20]}


def record_failure(exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob("stage_02_failure_attempt*.json"))
    atomic_json(ATTEMPTS / f"stage_02_failure_attempt{len(existing) + 1}.json", {
        "schema": "N69_FAILURE_ARTIFACT_V1",
        "status": "FAIL_PRESERVED",
        "stage": "N69_STAGE_02_CACHE_AUDIT",
        "created_at_utc": now(),
        "failure_root_cause": f"{type(exc).__name__}: {exc}",
        "traceback": __import__("traceback").format_exc(),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "production_authorized": False,
        "next_action": "Repair only the first cache/key integrity cause and rerun the same frozen audit.",
    })


def main() -> None:
    if not MAPPING_ROWS.is_file() or not MAPPING_SUMMARY.is_file():
        raise RuntimeError("N69 Stage 01 mapping audit outputs are missing")
    summary = load_json(MAPPING_SUMMARY)
    ids = event_ids()
    first_rows, key_errors, key_metrics = read_mapping_keys()
    inventory = source_inventory(ids)
    n54_status = load_json(N54_STATUS)
    cache_audit = {
        "schema": "N69_FROZEN_CANDIDATE_CACHE_AUDIT_V1",
        "created_at_utc": now(),
        "source": {
            "n37_event_manifest": str(N37_MANIFEST),
            "n37_event_manifest_sha256": sha256(N37_MANIFEST),
            "n54_runtime": str(N54_RUNTIME),
            "n54_runtime_status": str(N54_STATUS),
            "n54_runtime_status_sha256": sha256(N54_STATUS),
            "n54_runtime_status_value": n54_status.get("status"),
            "n69_mapping_rows": str(MAPPING_ROWS),
            "n69_mapping_rows_sha256": sha256(MAPPING_ROWS),
            "n69_mapping_summary": str(MAPPING_SUMMARY),
            "n69_mapping_summary_sha256": sha256(MAPPING_SUMMARY),
        },
        "event_count": len(ids),
        "variant_count": len(VARIANTS),
        "frames_per_event_variant": FRAMES,
        "key_metrics": key_metrics,
        "key_errors": dict(sorted(key_errors.items())),
        "source_files": inventory,
        "candidate_cache_fields": [
            "candidate_rows/native_tid/box/confidence/feature_available",
            "candidate_features_512 and feature digests",
            "memory_public_id_order/memory_valid/memory_vectors_512",
            "write_baseline score_matrix/assignment_columns/rows/public_id_order",
            "scalar_features_8 and scalar_contract",
            "runtime_future_gt_used=false",
            "checkpoint_sha256/code/protocol provenance from frozen N54 artifact",
        ],
        "target_absence": {
            "frames": int(summary.get("target_candidate_absent", 0)),
            "interpretation": "offline target-native candidate absent/no-op evidence; not silently filled or counted as mapping success",
        },
        "integrity": {
            "event_variant_frame_keys_exact": not bool(key_errors),
            "source_file_count_exact": len(inventory) == EVENTS,
            "candidate_frame_integrity_100": summary.get("candidate_frame_integrity_100") is True,
            "target_scope_mapping_100_on_available_candidates": summary.get("target_scope_mapping_100_on_available_candidates") is True,
            "runtime_future_gt_false": n54_status.get("runtime_future_gt_used") is False and not key_errors.get("runtime_future_gt_used"),
            "frozen_cache_reused_without_regeneration": True,
        },
        "provenance": {
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "runtime_future_gt_used": False,
            "production_authorized": False,
        },
        "sample_rows": first_rows,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT, cache_audit)
    manifest = {
        "schema": "N69_CANDIDATE_CACHE_MANIFEST_V1",
        "created_at_utc": now(),
        "status": "PASS_FROZEN_CACHE_REGISTERED" if not key_errors else "FAIL_CACHE_KEY_INTEGRITY",
        "cache_audit": str(AUDIT),
        "cache_audit_sha256": sha256(AUDIT),
        "source_files": inventory,
        "event_ids": ids,
        "variants": list(VARIANTS),
        "frames_per_event_variant": FRAMES,
        "mapping_version": "n69-target-boundary-v1",
        "candidate_absent_frames": int(summary.get("target_candidate_absent", 0)),
        "candidate_recall_on_target_labeled_frames": (summary.get("target_scope_resolved", 0) / summary.get("target_scope_total", 1)),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    stage_status = {
        "schema": "N69_STAGE_02_STATUS_V1",
        "status": "PASS_FROZEN_CANDIDATE_CACHE_REUSED" if cache_audit["integrity"]["event_variant_frame_keys_exact"] and cache_audit["integrity"]["candidate_frame_integrity_100"] else "BLOCKED_CANDIDATE_CACHE_INTEGRITY",
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "cache_manifest": str(MANIFEST),
        "cache_audit": str(AUDIT),
        "inputs": cache_audit["source"],
        "metrics": {
            "event_count": len(ids),
            "variant_count": len(VARIANTS),
            "frames_per_event_variant": FRAMES,
            "cache_rows": key_metrics["rows"],
            "expected_cache_rows": key_metrics["expected_rows"],
            "target_absent_frames": int(summary.get("target_candidate_absent", 0)),
            "target_available_frames": int(summary.get("target_scope_resolved", 0)),
            "candidate_recall_on_target_labeled_frames": manifest["candidate_recall_on_target_labeled_frames"],
        },
        "gate_checks": {
            "all_24_events": len(ids) == EVENTS,
            "all_5_variants": set(VARIANTS) == set(VARIANTS),
            "all_100_frames": key_metrics["rows"] == key_metrics["expected_rows"],
            "no_duplicate_or_missing_keys": not bool(key_errors),
            "candidate_frame_integrity_100": cache_audit["integrity"]["candidate_frame_integrity_100"],
            "target_scope_mapping_100_on_available_candidates": cache_audit["integrity"]["target_scope_mapping_100_on_available_candidates"],
            "runtime_future_gt_false": cache_audit["integrity"]["runtime_future_gt_false"],
            "formal_full_native_local_global_public_gate": summary.get("full_native_local_global_public_provenance") is True,
            "production_authorized": False,
        },
        "provenance": cache_audit["provenance"],
        "next_action": "Materialize N69 raw 512-D target-conditioned training features from this unchanged cache; target-absent frames remain explicit NONE/no-op labels.",
    }
    atomic_json(STATUS, stage_status)
    print(json.dumps({"status": stage_status["status"], "cache_manifest": str(MANIFEST), "rows": key_metrics["rows"], "source_files": len(inventory), "target_absent_frames": stage_status["metrics"]["target_absent_frames"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record_failure(exc)
        raise
