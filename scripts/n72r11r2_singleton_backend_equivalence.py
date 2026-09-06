#!/usr/bin/env python3
"""Audit the gated N72R11R2 singleton-backend route.

This audit is deliberately conservative: a singleton run that fails before
producing a complete target stream is not treated as equivalent.  The input
trees are read-only; the result is written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _done_path(root: Path, event_id: str) -> Path | None:
    direct = root / event_id / "done.json"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("done.json"))
    return matches[0] if len(matches) == 1 else None


def _failure_path(root: Path, event_id: str) -> Path | None:
    direct = root / "attempts" / f"{event_id}.failure.json"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(f"{event_id}.failure*.json"))
    return matches[0] if matches else None


def _runtime_config(done: dict[str, Any]) -> dict[str, Any]:
    policy = done.get("runtime_memory_policy")
    policy = policy if isinstance(policy, dict) else {}
    model_config = done.get("backend_model_config")
    model_config = model_config if isinstance(model_config, dict) else {}
    return {
        "max_num_objects": model_config.get("max_num_objects"),
        "multiplex_count": model_config.get("multiplex_count"),
        "official_trim_requested": policy.get("trim_past_non_cond_mem_for_eval"),
        "official_trim_enabled_frame_count": policy.get(
            "official_trim_enabled_frame_count"
        ),
        "streaming_propagation": done.get("target_session_audit", {}).get(
            "streaming_propagation_used"
        )
        if isinstance(done.get("target_session_audit"), dict)
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--singleton-root", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_done = _done_path(args.reference_root, args.event_id)
    singleton_done = _done_path(args.singleton_root, args.event_id)
    singleton_failure = _failure_path(args.singleton_root, args.event_id)
    result: dict[str, Any] = {
        "schema_version": "N72R11R2_SINGLETON_BACKEND_EQUIVALENCE_V1",
        "event_id": args.event_id,
        "protocol": {
            "reference": {"max_num_objects": 16, "multiplex_count": 16},
            "singleton": {"max_num_objects": 1, "multiplex_count": 1},
            "same_full_window": True,
            "horizon_override": None,
            "scientific_fields": [
                "frame_coverage",
                "candidate_presence",
                "candidate_kind",
                "source",
                "box_xyxy",
                "confidence",
                "presence_score",
                "mask_sha256",
            ],
        },
        "reference_root": str(args.reference_root.resolve()),
        "singleton_root": str(args.singleton_root.resolve()),
    }

    if reference_done is None:
        result["status"] = "BLOCKED_REFERENCE_ARTIFACT_MISSING"
        result["reason"] = "reference full-window run has no unique done.json"
        _atomic_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 2

    reference_payload = json.loads(reference_done.read_text(encoding="utf-8"))
    result["reference"] = {
        "done": str(reference_done.resolve()),
        "done_sha256": _sha256(reference_done),
        "status": reference_payload.get("status"),
        "frame_count": reference_payload.get("frame_count"),
        "runtime": _runtime_config(reference_payload),
    }

    if singleton_done is None:
        result["status"] = "SINGLETON_NOT_SEMANTICS_PRESERVING"
        result["reason"] = (
            "singleton configuration did not produce a complete done.json; "
            "the 1/1 checkpoint load failed before candidate generation"
        )
        if singleton_failure is not None:
            failure = json.loads(singleton_failure.read_text(encoding="utf-8"))
            result["singleton_failure"] = {
                "artifact": str(singleton_failure.resolve()),
                "artifact_sha256": _sha256(singleton_failure),
                "status": failure.get("status"),
                "failure_type": failure.get("failure_type"),
                "message": failure.get("message"),
                "traceback": failure.get("traceback"),
                "runtime_future_gt_used": failure.get("runtime_future_gt_used"),
            }
        else:
            result["singleton_failure"] = {
                "artifact": None,
                "reason": "no singleton done.json or failure artifact found",
            }
        result["decision"] = {
            "use_singleton_for_remaining_oom": False,
            "continue_to_resource_censored_route": True,
        }
        _atomic_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 2

    singleton_payload = json.loads(singleton_done.read_text(encoding="utf-8"))
    result["singleton"] = {
        "done": str(singleton_done.resolve()),
        "done_sha256": _sha256(singleton_done),
        "status": singleton_payload.get("status"),
        "frame_count": singleton_payload.get("frame_count"),
        "runtime": _runtime_config(singleton_payload),
    }

    # Delegate the exact candidate-content comparison to the shared audit.
    from n72r11r2_audit_streaming_equivalence import (  # pylint: disable=import-outside-toplevel
        _compare_rows,
        _load_rows,
        _resolve_stream,
    )

    comparison = _compare_rows(
        _load_rows(_resolve_stream(reference_done, reference_payload)),
        _load_rows(_resolve_stream(singleton_done, singleton_payload)),
    )
    result["scientific_comparison"] = comparison
    result["status"] = (
        "RESOURCE_EQUIVALENT_SINGLETON_BACKEND"
        if comparison["status"] == "PASS_SCIENTIFIC_CANDIDATE_EQUIVALENCE"
        else "SINGLETON_NOT_SEMANTICS_PRESERVING"
    )
    result["decision"] = {
        "use_singleton_for_remaining_oom": result["status"]
        == "RESOURCE_EQUIVALENT_SINGLETON_BACKEND",
        "continue_to_resource_censored_route": result["status"]
        != "RESOURCE_EQUIVALENT_SINGLETON_BACKEND",
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "RESOURCE_EQUIVALENT_SINGLETON_BACKEND" else 2


if __name__ == "__main__":
    raise SystemExit(main())
