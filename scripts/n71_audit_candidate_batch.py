#!/usr/bin/env python3
"""Audit all N71 candidate-window attempts without modifying their tapes."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from scripts.n71_audit_candidate_branch import audit_window, digest, load_event_manifest


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    events = load_event_manifest(args.event_manifest.resolve())
    results = []
    for item in payload["windows"]:
        root = args.smoke_root.resolve() if item["window_id"] == "n71-dancetrack0001-0296" else args.full_root.resolve()
        result = audit_window(item, root, events)
        result["root"] = str(root)
        results.append(result)
    failed = [r for r in results if r["status"] != "PASS"]
    summary = {
        "schema": "N71_CANDIDATE_BATCH_AUDIT_V1",
        "status": "PASS" if not failed and len(results) == len(payload["windows"]) else "FAIL",
        "plan": str(plan_path),
        "plan_sha256": digest(plan_path),
        "event_manifest": str(args.event_manifest.resolve()),
        "event_manifest_sha256": digest(args.event_manifest.resolve()),
        "window_count": len(results),
        "pass_count": sum(r["status"] == "PASS" for r in results),
        "fail_count": len(failed),
        "total_frame_count": sum(int(r["observed_frame_count"]) for r in results),
        "total_candidate_row_count": sum(int(r["candidate_row_count"]) for r in results),
        "total_degenerate_box_count_preserved": sum(int(r["degenerate_box_count_preserved"]) for r in results),
        "total_missing_mask_count": sum(int(r["missing_mask_count"]) for r in results),
        "runtime_future_gt_used_any": any(bool(r["runtime_future_gt_used"]) for r in results),
        "mapping_status_counts": {
            key: sum(int(r["candidate_mapping_statuses"].get(key, 0)) for r in results)
            for key in sorted({key for r in results for key in r["candidate_mapping_statuses"]})
        },
        "windows": results,
        "source_attempts_preserved": [
            "outputs/N71/attempts/stage_02_candidate_max32_checkpoint_mismatch.json",
            "outputs/N71/candidate_branch/smoke_audit_attempt2.json",
        ],
    }
    atomic_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
