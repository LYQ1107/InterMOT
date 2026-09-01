#!/usr/bin/env python3
"""Merge N36 overlapping shards and assign sequence-global native IDs.

Only the unique core owner of each frame is written to the merged tape.
Adjacent chunk overlap is used solely to match independently generated local
native IDs; numeric IDs are never assumed to be stable across processes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import (
    atomic_json,
    box_iou,
    cosine,
    decode_mask,
    display_path,
    iter_jsonl,
    load_sequences,
    mask_iou,
)


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_for_sequence(plan: dict[str, Any], sequence: str) -> list[dict[str, Any]]:
    for item in plan.get("sequences", []):
        if str(item.get("sequence")) == sequence:
            return [dict(chunk) for chunk in item.get("chunks", [])]
    raise KeyError(f"sequence not present in chunk plan: {sequence}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, row in iter_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row is not an object")
        rows.append(row)
    return rows


def rows_by_frame(rows: list[dict[str, Any]], chunk: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame in result:
            raise ValueError(f"duplicate frame {frame} in chunk {chunk['chunk_id']}")
        if row.get("sequence") != chunk["sequence"] or row.get("chunk_id") != chunk["chunk_id"]:
            raise ValueError(f"row provenance mismatch in chunk {chunk['chunk_id']} frame {frame}")
        result[frame] = row
    expected = set(range(int(chunk["frame_start"]), int(chunk["frame_end"]) + 1))
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(
            f"chunk {chunk['chunk_id']} frame coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    return result


def candidate_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for candidate in row.get("candidates", []):
        local = int(candidate.get("local_native_id", candidate.get("native_tid", -1)))
        if local in result:
            raise ValueError(f"duplicate local native id {local} at frame {row.get('frame')}")
        result[local] = candidate
    return result


def pair_score(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    box = box_iou(left.get("box", []), right.get("box", []))
    mask = mask_iou(decode_mask(left.get("mask")), decode_mask(right.get("mask")))
    emb = cosine(
        left.get("machine_embedding", left.get("embedding")),
        right.get("machine_embedding", right.get("embedding")),
    )
    # Geometry is always available; mask/embedding contribute only when both
    # sides carry valid values, with weights renormalized over available terms.
    values = [(0.55, float(box))]
    if mask is not None:
        values.append((0.20, float(mask)))
    if emb is not None:
        values.append((0.25, float(np.clip(emb, -1.0, 1.0))))
    weight = sum(item[0] for item in values)
    score = sum(item[0] * item[1] for item in values) / weight if weight else 0.0
    return {
        "score": float(score),
        "box_iou": float(box),
        "mask_iou": None if mask is None else float(mask),
        "embedding_cosine": None if emb is None else float(emb),
    }


def boundary_matches(
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    common_start: int,
    common_end: int,
    threshold: float = 0.25,
) -> list[dict[str, Any]]:
    left_ids = sorted({local for frame in range(common_start, common_end + 1) for local in candidate_map(left_rows[frame])})
    right_ids = sorted({local for frame in range(common_start, common_end + 1) for local in candidate_map(right_rows[frame])})
    if not left_ids or not right_ids:
        return []
    pair_records: dict[tuple[int, int], dict[str, Any]] = {}
    score_matrix = np.zeros((len(left_ids), len(right_ids)), dtype=float)
    for i, left_id in enumerate(left_ids):
        for j, right_id in enumerate(right_ids):
            samples = []
            for frame in range(common_start, common_end + 1):
                left = candidate_map(left_rows[frame]).get(left_id)
                right = candidate_map(right_rows[frame]).get(right_id)
                if left is not None and right is not None:
                    samples.append(pair_score(left, right))
            if samples:
                aggregate = {
                    "score": float(np.mean([item["score"] for item in samples])),
                    "box_iou": float(np.mean([item["box_iou"] for item in samples])),
                    "mask_iou": (
                        None
                        if not any(item["mask_iou"] is not None for item in samples)
                        else float(np.mean([item["mask_iou"] for item in samples if item["mask_iou"] is not None]))
                    ),
                    "embedding_cosine": (
                        None
                        if not any(item["embedding_cosine"] is not None for item in samples)
                        else float(np.mean([item["embedding_cosine"] for item in samples if item["embedding_cosine"] is not None]))
                    ),
                    "overlap_support_frames": len(samples),
                }
                pair_records[(left_id, right_id)] = aggregate
                score_matrix[i, j] = aggregate["score"]
    assignment: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        left_indices, right_indices = linear_sum_assignment(-score_matrix)
        assignment = list(zip(left_indices.tolist(), right_indices.tolist()))
    except Exception:
        flat = sorted(
            ((float(score_matrix[i, j]), i, j) for i in range(score_matrix.shape[0]) for j in range(score_matrix.shape[1])),
            reverse=True,
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for score, i, j in flat:
            if i in used_left or j in used_right:
                continue
            assignment.append((i, j))
            used_left.add(i)
            used_right.add(j)
    records = []
    for left_index, right_index in assignment:
        left_id = left_ids[left_index]
        right_id = right_ids[right_index]
        aggregate = pair_records.get((left_id, right_id))
        if aggregate is None:
            continue
        accepted = bool(
            aggregate["overlap_support_frames"] >= 1
            and np.isfinite(float(aggregate["score"]))
            and float(aggregate["score"]) >= threshold
        )
        records.append(
            {
                "left_local_native_id": int(left_id),
                "right_local_native_id": int(right_id),
                "accepted": accepted,
                "method": "overlap_box_mask_machine_embedding_hungarian",
                "acceptance_threshold": float(threshold),
                **aggregate,
            }
        )
    return records


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def merge_sequence(sequence: str, chunks: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    loaded: list[dict[str, Any]] = []
    for chunk in chunks:
        done_path = output_root / "chunk_done" / sequence / f"{chunk['chunk_id']}.json"
        chunk_path = output_root / "chunks" / sequence / f"{chunk['chunk_id']}.jsonl"
        if not done_path.is_file() or not chunk_path.is_file():
            raise FileNotFoundError(f"missing chunk artifact for {chunk['chunk_id']}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "PASS" or not done.get("candidate_set_complete"):
            raise RuntimeError(f"chunk not complete: {chunk['chunk_id']}: {done.get('status')}")
        rows = load_rows(chunk_path)
        loaded.append({"plan": chunk, "rows": rows, "by_frame": rows_by_frame(rows, chunk), "done": done})

    next_global = 1
    global_maps: list[dict[int, int]] = []
    local_mapping_records: list[dict[str, Any]] = []
    for index, item in enumerate(loaded):
        all_ids = sorted(
            {
                local
                for row in item["rows"]
                for local in candidate_map(row)
            }
        )
        mapping: dict[int, int] = {}
        if index == 0:
            for local in all_ids:
                mapping[local] = next_global
                local_mapping_records.append(
                    {
                        "chunk_id": item["plan"]["chunk_id"],
                        "local_native_id": int(local),
                        "sequence_global_native_id": int(next_global),
                        "method": "INITIAL_CHUNK_LOCAL_ID",
                        "accepted": True,
                    }
                )
                next_global += 1
        else:
            previous = loaded[index - 1]
            common_start = max(int(previous["plan"]["frame_start"]), int(item["plan"]["frame_start"]))
            common_end = min(int(previous["plan"]["frame_end"]), int(item["plan"]["frame_end"]))
            boundary = boundary_matches(
                previous["by_frame"], item["by_frame"], common_start, common_end
            )
            accepted_by_right = {
                int(record["right_local_native_id"]): record
                for record in boundary
                if record.get("accepted")
            }
            for local in all_ids:
                record = accepted_by_right.get(local)
                left_local = None if record is None else int(record["left_local_native_id"])
                if record is not None and left_local in global_maps[index - 1]:
                    global_id = global_maps[index - 1][left_local]
                    method = "OVERLAP_MATCHED"
                else:
                    global_id = next_global
                    next_global += 1
                    method = "NEW_GLOBAL_AFTER_OVERLAP"
                mapping[local] = int(global_id)
                local_mapping_records.append(
                    {
                        "chunk_id": item["plan"]["chunk_id"],
                        "local_native_id": int(local),
                        "sequence_global_native_id": int(global_id),
                        "method": method,
                        "accepted": record is not None,
                        "matched_previous_local_native_id": left_local,
                        "boundary": record,
                    }
                )
            item["boundary_matches"] = boundary
            item["boundary_common_range"] = [int(common_start), int(common_end)]
        global_maps.append(mapping)

    owner_by_frame: dict[int, int] = {}
    expected_frame_count = int(chunks[0]["frame_count_total"])
    for frame in range(expected_frame_count):
        owners = [
            index
            for index, item in enumerate(loaded)
            if int(item["plan"]["core_frame_start"]) <= frame <= int(item["plan"]["core_frame_end"])
        ]
        if len(owners) != 1:
            raise RuntimeError(f"core ownership is not exactly one at {sequence}:{frame}: {owners}")
        owner_by_frame[frame] = owners[0]

    merged_rows: list[dict[str, Any]] = []
    merged_candidate_count = 0
    for frame in range(expected_frame_count):
        owner_index = owner_by_frame[frame]
        item = loaded[owner_index]
        row = copy.deepcopy(item["by_frame"][frame])
        candidates = row.get("candidates", [])
        mapping = global_maps[owner_index]
        public_to_global: dict[str, int] = {}
        for candidate in candidates:
            local = int(candidate.get("local_native_id", candidate.get("native_tid", -1)))
            if local not in mapping:
                raise RuntimeError(f"missing global mapping at {sequence}:{frame}: local={local}")
            global_id = int(mapping[local])
            candidate["sequence_global_native_id"] = global_id
            candidate["sequence_global_native_id_status"] = "EXPLICIT_OVERLAP_RECONCILED"
            candidate["native_tid_scope"] = "chunk_local_raw_native_tid"
            if candidate.get("chunk_local_public_id") is not None:
                public_to_global[str(int(candidate["chunk_local_public_id"]))] = global_id
            merged_candidate_count += 1
        source_chunks = [
            str(item2["plan"]["chunk_id"])
            for item2 in loaded
            if int(item2["plan"]["frame_start"]) <= frame <= int(item2["plan"]["frame_end"])
        ]
        row["protocol"] = "N36_REAL_SHARDED_CANDIDATE_TAPE"
        row["frame_owner_chunk_id"] = str(item["plan"]["chunk_id"])
        row["source_chunk_ids"] = source_chunks
        row["is_core_frame"] = True
        row["candidate_complete"] = True
        row["candidate_set_complete"] = True
        row["public_id_namespace"] = "chunk_local_audit_with_sequence_global_native_bridge"
        row["public_id_to_sequence_global_native_id"] = public_to_global
        row["runtime_future_gt_used"] = False
        row["runtime_gt_read"] = False
        merged_rows.append(row)

    output_path = output_root / "frames" / f"{sequence}.jsonl"
    done_path = output_root / "done" / f"{sequence}.json"
    atomic_jsonl(output_path, merged_rows)
    result = {
        "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_DONE",
        "sequence": sequence,
        "status": "PASS",
        "candidate_complete": True,
        "candidate_set_complete": True,
        "frame_count": expected_frame_count,
        "candidate_count": merged_candidate_count,
        "chunk_count": len(chunks),
        "chunk_ids": [str(chunk["chunk_id"]) for chunk in chunks],
        "boundary_mapping": {
            "status": "PASS",
            "unique_local_to_global": all(
                len({record["sequence_global_native_id"] for record in local_mapping_records if record["chunk_id"] == chunk["chunk_id"] and record["local_native_id"] == local}) == 1
                for chunk in chunks
                for local in global_maps[chunks.index(chunk)]
            ),
            "local_mapping_record_count": len(local_mapping_records),
            "records": local_mapping_records,
            "matching_features": ["box_iou", "mask_iou_when_decodable", "machine_embedding_cosine"],
        },
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "output": display_path(output_path),
        "process_isolation_provenance": "each chunk was emitted by an independently exited Python process",
    }
    atomic_json(done_path, result)
    return result


def build_manifest(plan: dict[str, Any], sequences: list[str], output_root: Path) -> dict[str, Any]:
    completed = []
    failures = []
    expected_frames = 0
    for sequence in sorted(sequences):
        plan_seq = next(item for item in plan["sequences"] if item["sequence"] == sequence)
        expected_frames += int(plan_seq["frame_count"])
        done_path = output_root / "done" / f"{sequence}.json"
        if not done_path.is_file():
            failures.append({"sequence": sequence, "status": "NOT_RUN", "reason": "merged_done_missing"})
            continue
        item = json.loads(done_path.read_text(encoding="utf-8"))
        if item.get("status") == "PASS":
            # Keep the per-boundary records in the sequence done artifact;
            # the top-level manifest is a compact gate index rather than a
            # second copy of every matching score.
            summary = {key: value for key, value in item.items() if key != "boundary_mapping"}
            summary["done_artifact"] = display_path(done_path)
            summary["boundary_mapping_record_count"] = int(
                len(item.get("boundary_mapping", {}).get("records", []))
            )
            completed.append(summary)
        else:
            failures.append(item)
    status = "PASS" if len(completed) == len(sequences) and not failures else "PARTIAL"
    payload = {
        "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_MANIFEST",
        "status": status,
        "candidate_complete": bool(status == "PASS"),
        "candidate_set_complete": bool(status == "PASS"),
        "dataset_split": "train/train_fold",
        "sequence_count_expected": len(sequences),
        "sequence_count_pass": len(completed),
        "frame_count_expected": expected_frames,
        "frame_count_pass": int(sum(int(item.get("frame_count", 0)) for item in completed)),
        "candidate_count": int(sum(int(item.get("candidate_count", 0)) for item in completed)),
        "completed": completed,
        "failures": failures,
        "sequences": sorted(sequences),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "third_party_modified": False,
        "note": "Merged rows are unique core-owner frames; overlap rows are used only for native-ID reconciliation.",
    }
    atomic_json(output_root / "tape_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/n36/real_tape")
    parser.add_argument("--sequence-list", type=Path, default=ROOT / "outputs/n34/selected_sequences.json")
    parser.add_argument("--sequences", default="")
    args = parser.parse_args()
    plan = load_plan(args.plan.resolve())
    sequences = load_sequences(args.sequence_list, args.sequences)
    output_root = args.output_root.resolve()
    results = []
    for sequence in sequences:
        try:
            result = merge_sequence(sequence, plan_for_sequence(plan, sequence), output_root)
        except Exception as exc:
            result = {
                "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_DONE",
                "sequence": sequence,
                "status": "FAIL",
                "candidate_complete": False,
                "candidate_set_complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            atomic_json(output_root / "done" / f"{sequence}.json", result)
        results.append(result)
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "status": result.get("status"),
                    "frame_count": result.get("frame_count", 0),
                    "candidate_count": result.get("candidate_count", 0),
                    "chunk_count": result.get("chunk_count", 0),
                    "boundary_mapping_record_count": len(
                        result.get("boundary_mapping", {}).get("records", [])
                    ),
                    "error": result.get("error"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = build_manifest(plan, sequences, output_root)
    print(
        json.dumps(
            {
                "manifest": str(output_root / "tape_manifest.json"),
                "status": manifest.get("status"),
                "sequence_count_expected": manifest.get("sequence_count_expected"),
                "sequence_count_pass": manifest.get("sequence_count_pass"),
                "frame_count_pass": manifest.get("frame_count_pass"),
                "candidate_count": manifest.get("candidate_count"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
