#!/usr/bin/env python3
"""Audit scientific equivalence of two N72R11 secondary target streams.

The two runs intentionally use different SAM3 sessions.  This audit therefore
compares only candidate content and causal metadata, not session UUIDs or other
run-local identifiers.  It never edits either input tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


BOX_ATOL = 1e-4
BOX_RTOL = 1e-5
SCORE_ATOL = 1e-5


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_done(root: Path, event_id: str) -> Path:
    direct = root / event_id / "done.json"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("done.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one done.json below {root}, found {len(matches)}"
        )
    return matches[0]


def _resolve_stream(done_path: Path, done: dict[str, Any]) -> Path:
    declared = done.get("target_stream")
    candidates: list[Path] = []
    if isinstance(declared, str) and declared:
        candidates.extend((Path(declared), done_path.parent / declared))
    candidates.append(done_path.parent / "target_stream" / "frames.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"target stream not found for {done_path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _same_number(left: Any, right: Any, atol: float = SCORE_ATOL) -> bool:
    if _finite(left) and _finite(right):
        return abs(float(left) - float(right)) <= atol
    return left == right


def _same_box(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return left == right
    if len(left) != len(right):
        return False
    return all(
        _finite(a)
        and _finite(b)
        and math.isclose(float(a), float(b), rel_tol=BOX_RTOL, abs_tol=BOX_ATOL)
        for a, b in zip(left, right)
    )


def _candidate_comparison(
    reference: dict[str, Any], candidate: dict[str, Any], frame: Any, index: int
) -> tuple[list[str], list[str]]:
    scientific_mismatches: list[str] = []
    mask_mismatches: list[str] = []
    if reference.get("candidate_kind") != candidate.get("candidate_kind"):
        scientific_mismatches.append(f"frame={frame} candidate={index} candidate_kind")
    if reference.get("source") != candidate.get("source"):
        scientific_mismatches.append(f"frame={frame} candidate={index} source")
    if not _same_box(reference.get("box_xyxy"), candidate.get("box_xyxy")):
        scientific_mismatches.append(f"frame={frame} candidate={index} box_xyxy")
    for field in ("confidence", "presence_score"):
        if not _same_number(reference.get(field), candidate.get(field)):
            scientific_mismatches.append(f"frame={frame} candidate={index} {field}")
    if reference.get("mask_sha256") != candidate.get("mask_sha256"):
        mask_mismatches.append(f"frame={frame} candidate={index} mask_sha256")
    return scientific_mismatches, mask_mismatches


def _compare_rows(
    reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    mismatches: list[str] = []
    mask_mismatches: list[str] = []
    if [row.get("frame") for row in reference_rows] != [
        row.get("frame") for row in candidate_rows
    ]:
        mismatches.append("frame_coverage")
    if len(reference_rows) != len(candidate_rows):
        mismatches.append("row_count")

    for row_index, (reference, candidate) in enumerate(
        zip(reference_rows, candidate_rows)
    ):
        frame = reference.get("frame")
        if reference.get("candidate_count") != candidate.get("candidate_count"):
            mismatches.append(f"frame={frame} candidate_count")
        if reference.get("candidate_set_complete") != candidate.get(
            "candidate_set_complete"
        ):
            mismatches.append(f"frame={frame} candidate_set_complete")
        for field in (
            "is_event_frame",
            "is_future_frame",
            "event_frame_memory_read",
            "runtime_future_gt_used",
            "runtime_gt_read",
        ):
            if reference.get(field) != candidate.get(field):
                mismatches.append(f"frame={frame} {field}")
        reference_candidates = reference.get("candidate_rows") or []
        candidate_candidates = candidate.get("candidate_rows") or []
        if len(reference_candidates) != len(candidate_candidates):
            mismatches.append(f"frame={frame} candidate_presence")
        for candidate_index, (ref_item, candidate_item) in enumerate(
            zip(reference_candidates, candidate_candidates)
        ):
            scientific, masks = _candidate_comparison(
                ref_item, candidate_item, frame, candidate_index
            )
            mismatches.extend(scientific)
            mask_mismatches.extend(masks)

    unique_mismatches = sorted(set(mismatches))
    unique_mask_mismatches = sorted(set(mask_mismatches))
    if unique_mismatches:
        status = "FAIL_SCIENTIFIC_CANDIDATE_MISMATCH"
    elif unique_mask_mismatches:
        status = "MASK_HASH_NONDETERMINISM"
    else:
        status = "PASS_SCIENTIFIC_CANDIDATE_EQUIVALENCE"
    return {
        "status": status,
        "reference_row_count": len(reference_rows),
        "candidate_row_count": len(candidate_rows),
        "reference_frame_count": len({row.get("frame") for row in reference_rows}),
        "candidate_frame_count": len({row.get("frame") for row in candidate_rows}),
        "scientific_mismatch_count": len(unique_mismatches),
        "scientific_mismatch_examples": unique_mismatches[:50],
        "mask_hash_mismatch_count": len(unique_mask_mismatches),
        "mask_hash_mismatch_examples": unique_mask_mismatches[:50],
    }


def _runtime_summary(done: dict[str, Any]) -> dict[str, Any]:
    policy = done.get("runtime_memory_policy")
    session = done.get("target_session_audit")
    policy = policy if isinstance(policy, dict) else {}
    session = session if isinstance(session, dict) else {}
    blockers = policy.get("official_trim_schema_blockers")
    blockers = blockers if isinstance(blockers, list) else []
    return {
        "status": done.get("status"),
        "attempt": done.get("attempt"),
        "frame_count": done.get("frame_count"),
        "streaming_propagation_used": session.get("streaming_propagation_used"),
        "streaming_backend_cache_outputs": session.get(
            "streaming_backend_cache_outputs"
        ),
        "official_trim_requested": policy.get("trim_past_non_cond_mem_for_eval"),
        "official_trim_targets": policy.get("trim_past_non_cond_mem_targets", []),
        "official_trim_enabled_frame_count": policy.get(
            "official_trim_enabled_frame_count"
        ),
        "official_trim_schema_blocker_count": len(blockers),
        "official_trim_schema_blocker_reasons": sorted(
            {item.get("reason") for item in blockers if isinstance(item, dict)}
        ),
        "offload_output_to_cpu_for_eval": policy.get("offload_output_to_cpu_for_eval"),
        "offload_state_to_cpu": policy.get("offload_state_to_cpu"),
        "offload_video_to_cpu": policy.get("offload_video_to_cpu"),
        "runtime_future_gt_used": done.get("runtime_future_gt_used"),
        "runtime_gt_read": done.get("runtime_gt_read"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_done_path = _find_done(args.reference_root, args.event_id)
    candidate_done_path = _find_done(args.candidate_root, args.event_id)
    reference_done = json.loads(reference_done_path.read_text(encoding="utf-8"))
    candidate_done = json.loads(candidate_done_path.read_text(encoding="utf-8"))
    reference_stream = _resolve_stream(reference_done_path, reference_done)
    candidate_stream = _resolve_stream(candidate_done_path, candidate_done)
    comparison = _compare_rows(
        _load_rows(reference_stream), _load_rows(candidate_stream)
    )
    result = {
        "schema_version": "N72R11R2_STREAMING_EQUIVALENCE_AUDIT_V1",
        "event_id": args.event_id,
        "scientific_comparison": comparison,
        "reference": {
            "done": str(reference_done_path.resolve()),
            "done_sha256": _sha256(reference_done_path),
            "stream": str(reference_stream),
            "stream_sha256": _sha256(reference_stream),
            "runtime": _runtime_summary(reference_done),
        },
        "candidate": {
            "done": str(candidate_done_path.resolve()),
            "done_sha256": _sha256(candidate_done_path),
            "stream": str(candidate_stream),
            "stream_sha256": _sha256(candidate_stream),
            "runtime": _runtime_summary(candidate_done),
        },
        "interpretation": {
            "session_identifiers_compared": False,
            "scientific_candidate_content_compared": True,
            "mask_hash_mismatch_is_not_auto_pass": True,
            "official_trim_runtime_effective": (
                candidate_done.get("runtime_memory_policy", {}).get(
                    "official_trim_enabled_frame_count", 0
                )
                > 0
            ),
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if comparison["status"] == "PASS_SCIENTIFIC_CANDIDATE_EQUIVALENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
