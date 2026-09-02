#!/usr/bin/env python3
"""Prepare and audit the N72R2 independent-session handover ladder.

The official SAM3 exporter is intentionally kept in ``n72r2_stage01``.  This
module only creates a deterministic overlap plan and reconciles two completed
same-sequence sessions using overlap observations.  It never imports GT and it
never treats a raw SAM id or a newly allocated session-local MOT id as a
sequence-global public id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.identity.handover import PersistentLineageHandover


N72R2_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2")
FROZEN_PLAN = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def read_run_metadata(output_root: Path) -> dict[str, Any]:
    """Read same-run session/segment metadata from the immutable V2 rows."""

    rows = read_jsonl(output_root / "candidate_v2.jsonl")
    if not rows:
        raise ValueError(f"candidate_v2 is empty: {output_root}")
    first = rows[0]
    required = ("source_run_id", "session_id", "segment_id", "sequence", "window_id")
    missing = [key for key in required if first.get(key) is None]
    if missing:
        raise ValueError(f"candidate_v2 metadata missing {missing}: {output_root}")
    return {key: first[key] for key in required}


def load_window(path: Path, window_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("windows", []):
        if str(item.get("window_id")) == window_id:
            return dict(item)
    raise KeyError(f"window {window_id!r} is absent from {path}")


def prepare_overlap_plan(output: Path) -> dict[str, Any]:
    """Create a new N72R2 plan without modifying the N71 frozen plan."""

    base = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    first = load_window(FROZEN_PLAN, "n71-dancetrack0001-0296")
    overlap = 20
    length = int(base.get("settings", {}).get("chunk_frames", 160))
    next_start = int(first["frame_end"]) - overlap + 1
    next_end = next_start + length - 1
    second = {
        "action_type": "STRUCTURAL_HANDOVER_AUDIT",
        "core_end": next_end - overlap,
        "core_start": next_start + overlap,
        "event_id": None,
        "frame_count_total": int(first["frame_count_total"]),
        "frame_end": next_end,
        "frame_start": next_start,
        "future_core_frames": None,
        "interaction_source": "none_structural_handover_only",
        "prefix_overlap_frames": overlap,
        "runtime_future_gt_used": False,
        "selection_basis": "deterministic_same_sequence_adjacent_range_from_stage01_smoke; no GT or replay fields",
        "sequence": str(first["sequence"]),
        "window_id": "n72r2-dancetrack0001-overlap-0416",
    }
    if next_end >= int(first["frame_count_total"]):
        raise ValueError(f"deterministic overlap range exceeds sequence: {next_start}:{next_end}")
    plan = {
        "schema": "N72R2_HANDOVER_OVERLAP_PLAN_V1",
        "created_from": str(FROZEN_PLAN),
        "created_from_sha256": sha256(FROZEN_PLAN),
        "status": "FROZEN_BEFORE_SECOND_SESSION",
        "settings": {
            "chunk_frames": length,
            "overlap_frames": overlap,
            "runtime_future_gt_used": False,
        },
        "selection_basis": "first N72R2 Stage01 range plus deterministic adjacent same-sequence range; no GT/post-treatment inputs",
        "windows": [first, second],
    }
    atomic_json(output, plan)
    return plan


def _validate_rows(rows: list[dict[str, Any]], label: str) -> tuple[set[int], set[int], set[tuple[int, str]]]:
    frames: set[int] = set()
    tracks: set[int] = set()
    keys: set[tuple[int, str]] = set()
    for row in rows:
        for key in ("frame_idx", "mot_track_id", "public_id", "lineage_id", "box", "feature"):
            if key not in row:
                raise ValueError(f"{label} row missing {key}: {row}")
        frame = int(row["frame_idx"])
        track = int(row["mot_track_id"])
        if row.get("public_id") is None:
            raise ValueError(f"{label} row has no same-run public authority: {row}")
        key = (frame, str(row.get("candidate_uid")))
        if key in keys:
            raise ValueError(f"duplicate {label} candidate key: {key}")
        keys.add(key)
        frames.add(frame)
        tracks.add(track)
    if not rows:
        raise ValueError(f"{label} rows are empty")
    return frames, tracks, keys


def audit_overlap(
    previous_output: Path,
    next_output: Path,
    output_root: Path,
    *,
    source_run_id: str,
    previous_window_id: str,
    next_window_id: str,
    overlap_start: int,
    overlap_end: int,
) -> dict[str, Any]:
    previous = read_jsonl(previous_output / "public_mot_tracks.jsonl")
    following = read_jsonl(next_output / "public_mot_tracks.jsonl")
    previous_meta = read_run_metadata(previous_output)
    next_meta = read_run_metadata(next_output)
    if str(previous_meta["sequence"]) != str(next_meta["sequence"]):
        raise ValueError("cross-sequence handover is not allowed")
    prev_frames, prev_tracks, _ = _validate_rows(previous, "previous")
    next_frames, next_tracks, _ = _validate_rows(following, "next")
    expected_overlap = set(range(int(overlap_start), int(overlap_end) + 1))
    prev_overlap = [row for row in previous if int(row["frame_idx"]) in expected_overlap]
    next_overlap = [row for row in following if int(row["frame_idx"]) in expected_overlap]
    prev_overlap_frames = {int(row["frame_idx"]) for row in prev_overlap}
    next_overlap_frames = {int(row["frame_idx"]) for row in next_overlap}
    if prev_overlap_frames != expected_overlap or next_overlap_frames != expected_overlap:
        raise ValueError(
            "overlap frame coverage mismatch: "
            f"previous_missing={sorted(expected_overlap - prev_overlap_frames)}, "
            f"next_missing={sorted(expected_overlap - next_overlap_frames)}"
        )
    handover = PersistentLineageHandover(source_run_id, "dancetrack0001")
    transactions = handover.match_overlap(
        prev_overlap,
        next_overlap,
        from_session=str(previous_meta["session_id"]),
        to_session=str(next_meta["session_id"]),
        from_segment=str(previous_meta["segment_id"]),
        to_segment=str(next_meta["segment_id"]),
        frame_boundary=int(overlap_start),
        min_iou=0.20,
        min_score=0.20,
    )
    old_by_track = {int(row["mot_track_id"]): row for row in prev_overlap}
    new_by_track = {int(row["mot_track_id"]): row for row in next_overlap}
    old_overlap_tracks = set(old_by_track)
    new_overlap_tracks = set(new_by_track)
    mapped_old = {int(tx.old_adapter_id) for tx in transactions}
    mapped_new = {int(tx.new_adapter_id) for tx in transactions}
    # Adapter IDs are stable within a session; completeness is assessed by
    # track IDs below, while the adapter sets remain useful diagnostics.
    mapped_old_tracks = {
        int(old_by_track[tid]["mot_track_id"])
        for tid in old_overlap_tracks
        if int(old_by_track[tid].get("adapter_external_id", -1)) in mapped_old
    }
    mapped_new_tracks = {
        int(new_by_track[tid]["mot_track_id"])
        for tid in new_overlap_tracks
        if int(new_by_track[tid].get("adapter_external_id", -1)) in mapped_new
    }
    public_ids = [int(tx.public_id) for tx in transactions]
    output_tracks: list[dict[str, Any]] = []
    transaction_by_new_adapter = {int(tx.new_adapter_id): tx for tx in transactions}
    for row in following:
        new_adapter = int(row.get("adapter_external_id", -1))
        tx = transaction_by_new_adapter.get(new_adapter)
        item = dict(row)
        if tx is None:
            item["sequence_public_id"] = None
            item["handover_status"] = "NO_OVERLAP_TRANSACTION"
        else:
            item["sequence_public_id"] = int(tx.public_id)
            item["handover_status"] = "PASS"
            item["handover_transaction_id"] = (
                f"{tx.source_run_id}:{tx.from_segment}->{tx.to_segment}:{tx.public_id}"
            )
        item["runtime_future_gt_used"] = False
        output_tracks.append(item)
    audit = handover.audit(expected_pairs=len(old_overlap_tracks))
    audit.update(
        {
            "schema_version": "N72R2_HANDOVER_OVERLAP_AUDIT_V1",
            "previous_window_id": previous_window_id,
            "next_window_id": next_window_id,
            "previous_source_run_id": str(previous_meta["source_run_id"]),
            "next_source_run_id": str(next_meta["source_run_id"]),
            "previous_session_id": str(previous_meta["session_id"]),
            "next_session_id": str(next_meta["session_id"]),
            "previous_segment_id": str(previous_meta["segment_id"]),
            "next_segment_id": str(next_meta["segment_id"]),
            "overlap_start": int(overlap_start),
            "overlap_end": int(overlap_end),
            "overlap_frame_count": len(expected_overlap),
            "previous_frame_count": len(prev_frames),
            "next_frame_count": len(next_frames),
            "previous_track_count": len(prev_tracks),
            "next_track_count": len(next_tracks),
            "overlap_previous_track_count": len(old_overlap_tracks),
            "overlap_next_track_count": len(new_overlap_tracks),
            "mapped_previous_track_count": len(mapped_old_tracks),
            "mapped_next_track_count": len(mapped_new_tracks),
            "overlap_mapping_coverage": (
                min(len(mapped_old_tracks), len(mapped_new_tracks)) / len(old_overlap_tracks)
                if old_overlap_tracks
                else 0.0
            ),
            "adapter_ids_in_transactions": len(mapped_old),
            "public_id_collision_count": len(public_ids) - len(set(public_ids)),
            "raw_id_equality_used_for_match": False,
            "runtime_future_gt_used": False,
            "status": (
                "PASS_EXPLICIT_OVERLAP_HANDOVER"
                if old_overlap_tracks
                and old_overlap_tracks == mapped_old_tracks
                and new_overlap_tracks == mapped_new_tracks
                and len(public_ids) == len(set(public_ids))
                else "PARTIAL_HANDOVER"
            ),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "handover_transactions.jsonl", [tx.as_dict() for tx in transactions])
    atomic_jsonl(output_root / "next_window_reconciled_public_tracks.jsonl", output_tracks)
    atomic_json(output_root / "handover_gate.json", audit)
    atomic_json(
        output_root / "input_manifest.json",
        {
            "previous_output": str(previous_output),
            "previous_public_tracks_sha256": sha256(previous_output / "public_mot_tracks.jsonl"),
            "next_output": str(next_output),
            "next_public_tracks_sha256": sha256(next_output / "public_mot_tracks.jsonl"),
            "previous_candidate_v2_sha256": sha256(previous_output / "candidate_v2.jsonl"),
            "next_candidate_v2_sha256": sha256(next_output / "candidate_v2.jsonl"),
            "runtime_future_gt_used": False,
        },
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--previous-output", type=Path, required=True)
    audit.add_argument("--next-output", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--source-run-id", required=True)
    audit.add_argument("--previous-window-id", required=True)
    audit.add_argument("--next-window-id", required=True)
    audit.add_argument("--overlap-start", type=int, required=True)
    audit.add_argument("--overlap-end", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_overlap_plan(args.output)
        print(json.dumps({"status": payload["status"], "windows": len(payload["windows"]), "output": str(args.output)}))
    else:
        result = audit_overlap(
            args.previous_output,
            args.next_output,
            args.output_root,
            source_run_id=str(args.source_run_id),
            previous_window_id=str(args.previous_window_id),
            next_window_id=str(args.next_window_id),
            overlap_start=int(args.overlap_start),
            overlap_end=int(args.overlap_end),
        )
        print(json.dumps({"status": result["status"], "transactions": result["transaction_count"], "coverage": result["overlap_mapping_coverage"]}))


if __name__ == "__main__":
    main()
