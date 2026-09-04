#!/usr/bin/env python3
"""CPU-only integrity audit for N72R7-R5 re-query artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit(root: Path, *, attempt: int, run_label: str = "auto") -> dict[str, Any]:
    batch_path = root / f"batch_attempt{int(attempt)}.json"
    batch = read_json(batch_path)
    errors: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_frames = 0
    for result in batch.get("results", []):
        event_id = str(result["event_id"])
        event_root = root / f"attempt_{int(attempt)}" / event_id
        done_path = event_root / "done.json"
        frames_path = event_root / "frames.jsonl"
        if not done_path.is_file() or not frames_path.is_file():
            errors.append({"event_id": event_id, "reason": "missing_done_or_frames"})
            continue
        done = read_json(done_path)
        rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if done.get("status") != "PASS_N72R7_CANDIDATE_GENERATOR_REQUERY":
            errors.append({"event_id": event_id, "reason": "done_not_pass"})
        if len(rows) != 101 or done.get("frame_count") != 101:
            errors.append({"event_id": event_id, "reason": "frame_count", "value": len(rows)})
        if rows:
            event_frame = int(rows[0]["event_frame"])
            expected = list(range(event_frame, event_frame + 101))
            if [int(row["frame"]) for row in rows] != expected:
                errors.append({"event_id": event_id, "reason": "global_frame_axis"})
        query_audits = done.get("query_audits", [])
        if len(query_audits) != 5 or any(item.get("status") != "PASS_QUERY" for item in query_audits):
            errors.append({"event_id": event_id, "reason": "query_audit_incomplete"})
        event_rows = 0
        min_count = None
        max_count = None
        for row in rows:
            frame = int(row["frame"])
            candidates = list(row.get("candidate_rows", []))
            if int(row.get("candidate_count", -1)) != len(candidates):
                errors.append({"event_id": event_id, "frame": frame, "reason": "candidate_count"})
            uids = [str(item.get("candidate_uid")) for item in candidates]
            if len(uids) != len(set(uids)):
                errors.append({"event_id": event_id, "frame": frame, "reason": "duplicate_candidate_uid"})
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                if row.get(flag) is not False:
                    errors.append({"event_id": event_id, "frame": frame, "reason": f"row_{flag}"})
            for candidate in candidates:
                if candidate.get("candidate_source") != "TARGET_SESSION_REQUERY":
                    errors.append({"event_id": event_id, "frame": frame, "reason": "wrong_candidate_source"})
                if candidate.get("candidate_kind") != "TARGET_CORRECTION_SESSION_REQUERY_CANDIDATE":
                    errors.append({"event_id": event_id, "frame": frame, "reason": "wrong_candidate_kind"})
                if candidate.get("public_id") is not None or candidate.get("public_id_inference") is not False:
                    errors.append({"event_id": event_id, "frame": frame, "reason": "candidate_public_authority"})
                feature = candidate.get("feature")
                if not isinstance(feature, list) or len(feature) != 512 or not all(math.isfinite(float(value)) for value in feature):
                    errors.append({"event_id": event_id, "frame": frame, "reason": "feature_invalid"})
                elif hashlib.sha256(np.asarray(feature, dtype="<f4").tobytes()).hexdigest() != str(candidate.get("feature_sha256")):
                    errors.append({"event_id": event_id, "frame": frame, "reason": "feature_hash"})
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                    if candidate.get(flag) is not False:
                        errors.append({"event_id": event_id, "frame": frame, "reason": f"candidate_{flag}"})
            count = len(candidates)
            event_rows += count
            min_count = count if min_count is None else min(min_count, count)
            max_count = count if max_count is None else max(max_count, count)
        if rows:
            if int(rows[0]["candidate_count"]) <= 0:
                errors.append({"event_id": event_id, "reason": "event_frame_has_no_query_candidate"})
            if int(rows[1]["frame"]) != int(rows[0]["event_frame"]) + 1:
                errors.append({"event_id": event_id, "reason": "event_plus_one_missing"})
        total_rows += event_rows
        total_frames += len(rows)
        event_summaries.append({
            "event_id": event_id,
            "sequence": done.get("sequence"),
            "action_type": done.get("action_type"),
            "frame_count": len(rows),
            "candidate_row_count": event_rows,
            "min_candidates_per_frame": min_count,
            "max_candidates_per_frame": max_count,
            "query_count": len(query_audits),
        })
    passed = not errors and not batch.get("failures") and len(event_summaries) == int(batch.get("requested_event_count", -1))
    if run_label == "auto":
        run_label = "smoke" if int(batch.get("requested_event_count", -1)) == 3 else "full"
    if run_label not in {"smoke", "full"}:
        raise ValueError(f"unsupported run label: {run_label}")
    status_suffix = "SMOKE" if run_label == "smoke" else "FULL_AUDIT"
    return {
        "schema_version": "N72R7_CANDIDATE_GENERATOR_REQUERY_AUDIT_V1",
        "status": f"{'PASS' if passed else 'FAIL'}_N72R7_R5_REQUERY_{status_suffix}",
        "run_label": run_label,
        "input_batch": str(batch_path),
        "attempt": int(attempt),
        "event_count": len(event_summaries),
        "frame_count": total_frames,
        "candidate_row_count": total_rows,
        "query_count_per_event": 5,
        "errors": errors,
        "batch_failures": batch.get("failures", []),
        "events": event_summaries,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", choices=("auto", "smoke", "full"), default="auto")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit(root, attempt=int(args.attempt), run_label=str(args.run_label))
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "event_count": result["event_count"], "candidate_row_count": result["candidate_row_count"], "errors": len(result["errors"])}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
