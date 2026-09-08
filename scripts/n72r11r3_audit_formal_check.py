#!/usr/bin/env python3
"""Audit the single authorized N72R11R3 formal resource-validation event."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = "n72r5-pool-n37-dancetrack0062-0291-add_new_identity-001"
EXPECTED_VARIANTS = ("E0_BASELINE_B0", "E1_V3_LEGACY_INJECTION", "E2_V3_CORRECTED_BRIDGE", "E3_V3_CORRECTED_BRIDGE_LIVE")


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def walk_contract(value: Any, location: str, errors: list[str]) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_identity_error", "future_iou", "future_gt", "gt_target", "gt_id"}
    flags = {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key).lower()
            if text in forbidden:
                # Posthoc payloads are allowed to contain offline GT fields, but
                # runtime frame/manifest rows are not.
                if "/posthoc" not in location and "posthoc" not in text:
                    errors.append(f"{location}/{key}:forbidden_runtime_field")
            if text in flags and text != "posthoc_gt_used" and nested is not False:
                errors.append(f"{location}/{key}:not_false")
            walk_contract(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            walk_contract(nested, f"{location}/{index}", errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    event_root = args.event_root if args.event_root.is_absolute() else ROOT / args.event_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    errors: list[str] = []
    done_path = event_root / "done.json"
    done = read_json(done_path)
    if str(done.get("event_id")) != EVENT_ID:
        errors.append(f"event_id mismatch: {done.get('event_id')!r}")
    if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
        errors.append(f"done status: {done.get('status')!r}")
    if int(done.get("horizon", -1)) != 100:
        errors.append(f"horizon: {done.get('horizon')!r}")
    for key in ("runtime_future_gt_used", "runtime_gt_read"):
        if key in done and done.get(key) is not False:
            errors.append(f"done {key} is not false")
    manifest_path = event_root / "runtime_event_sealed.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R11_ALL_RUNTIME_SEALED":
        errors.append(f"seal status: {manifest.get('status')!r}")
    if set(manifest.get("variants", [])) != set(EXPECTED_VARIANTS):
        errors.append(f"variant set: {manifest.get('variants')!r}")
    for variant in EXPECTED_VARIANTS:
        runtime_manifest_path = event_root / variant / "runtime_manifest.json"
        frames_path = event_root / variant / "runtime_frames.jsonl"
        runtime_manifest = read_json(runtime_manifest_path)
        if runtime_manifest.get("status") != "PASS_N72R11_RUNTIME_ARTIFACT_SEALED":
            errors.append(f"{variant} manifest not sealed")
        if runtime_manifest.get("frames_sha256") != sha256_file(frames_path):
            errors.append(f"{variant} frame digest mismatch")
        rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 101:
            errors.append(f"{variant} frame count {len(rows)} != 101")
        if not rows or "frame" not in rows[0]:
            errors.append(f"{variant} event frame row missing")
        event_frame = int(rows[0].get("frame", -1)) if rows else -1
        if rows and rows[0].get("memory_read") is not False:
            errors.append(f"{variant} event frame memory read is not false")
        if rows and int(rows[0].get("first_memory_visible_frame", -1)) != event_frame + 1:
            errors.append(f"{variant} first visible frame boundary mismatch")
        if rows and int(rows[-1].get("frame", -1)) != event_frame + 100:
            errors.append(f"{variant} H100 suffix is incomplete")
        walk_contract(runtime_manifest, f"{variant}/manifest", errors)
        for index, row in enumerate(rows):
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                errors.append(f"{variant}/row{index}:runtime GT contract")
            if row.get("public_id_inference") is not False:
                errors.append(f"{variant}/row{index}:public-ID inference")
            if not isinstance(row.get("candidate_rows"), list):
                errors.append(f"{variant}/row{index}:candidate rows missing")
        walk_contract(rows, f"{variant}/rows", errors)
    posthoc_path = event_root / "posthoc.json"
    posthoc = read_json(posthoc_path)
    if posthoc.get("status") != "PASS_N72R11_POSTHOC_EVENT" or posthoc.get("runtime_future_gt_used") is not False or posthoc.get("posthoc_gt_used") is not True:
        errors.append("posthoc provenance/status invalid")
    controller = done.get("stats", {}).get("E3_V3_CORRECTED_BRIDGE_LIVE", {}).get("controller_audit", {})
    if controller.get("post_session_feature_materializer") is not True:
        errors.append("post-session feature materializer was not enabled")
    if controller.get("feature_materialization_phase") != "after_sam3_session_release":
        errors.append("feature materialization phase was not after SAM3 release")
    event_frame = int(json.loads((event_root / EXPECTED_VARIANTS[0] / "runtime_frames.jsonl").read_text(encoding="utf-8").splitlines()[0]).get("frame", -1))
    if controller.get("event_frame_memory_read") is not False or controller.get("first_memory_visible_frame") != event_frame + 1:
        errors.append("controller causal boundary invalid")
    status = "PASS_N72R11R3_FORMAL_OOM_RESOURCE_CHECK" if not errors else "FAIL_N72R11R3_FORMAL_OOM_RESOURCE_CHECK_AUDIT"
    result = {
        "schema_version": "N72R11R3_FORMAL_RESOURCE_AUDIT_V1",
        "status": status,
        "created_at_utc": now_utc(),
        "event_id": EVENT_ID,
        "event_root": str(event_root),
        "done": str(done_path),
        "done_sha256": sha256_file(done_path),
        "runtime_event_sealed": str(manifest_path),
        "runtime_event_sealed_sha256": sha256_file(manifest_path),
        "posthoc": str(posthoc_path),
        "posthoc_sha256": sha256_file(posthoc_path),
        "event_frame": event_frame,
        "horizon": 100,
        "variants": list(EXPECTED_VARIANTS),
        "post_session_feature_materializer": True,
        "feature_materialization_phase": "after_sam3_session_release",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "oom_observed": False,
        "errors": errors,
        "resource_validation_scope": "single_original_formal_oom_event_full_window_only; no scientific promotion",
        "finished_at_utc": now_utc(),
    }
    atomic_json(output, result)
    print(json.dumps({"status": status, "output": str(output), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
