#!/usr/bin/env python3
"""CPU-only integrity and posthoc recall audit for an N71 candidate window.

The exporter deliberately has no annotation or public-ID input.  This audit is
allowed to read the frozen event/GT files only after export, and records that
separation explicitly; it never writes a mapping back into the candidate tape.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def decode_mask(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("encoding") != "packbits_zlib_base64":
        return None
    try:
        shape = tuple(int(v) for v in payload["shape"])
        if len(shape) != 2 or min(shape) <= 0:
            return None
        raw = zlib.decompress(base64.b64decode(str(payload["data"]).encode("ascii")))
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        count = shape[0] * shape[1]
        if bits.size < count:
            return None
        arr = bits[:count].reshape(shape).astype(bool, copy=False)
        declared = str(payload.get("sha256", ""))
        if declared and hashlib.sha256(arr.tobytes()).hexdigest() != declared:
            return None
        return arr.copy()
    except (TypeError, ValueError, KeyError, zlib.error):
        return None


def box_iou(a: Any, b: Any) -> float:
    try:
        x = np.asarray(a, dtype=np.float64).reshape(4)
        y = np.asarray(b, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return 0.0
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return 0.0
    ix1, iy1 = max(x[0], y[0]), max(x[1], y[1])
    ix2, iy2 = min(x[2], y[2]), min(x[3], y[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_x = max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])
    area_y = max(0.0, y[2] - y[0]) * max(0.0, y[3] - y[1])
    union = area_x + area_y - inter
    return float(inter / union) if union > 0 else 0.0


def load_gt_boxes(sequence: str, gt_id: int) -> dict[int, list[float]]:
    path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
    out: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = [v.strip() for v in line.split(",")]
            if len(fields) < 6:
                raise ValueError(f"malformed GT line {path}:{line_no}")
            # DanceTrack annotation frames are one-based; SAM3 tape frames are
            # zero-based image-list indices, matching N36/N70.
            frame, identity = int(fields[0]) - 1, int(fields[1])
            if identity != int(gt_id):
                continue
            x, y, w, h = (float(v) for v in fields[2:6])
            out[frame] = [x, y, x + w, y + h]
    return out


def load_event_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in payload.get("events", []):
        event = item.get("event", item)
        result[str(event["event_id"])] = {"event": event, "sequence_frame_count": int(item["sequence_frame_count"])}
    return result


def audit_window(plan_item: dict[str, Any], output_root: Path, events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    window_id = str(plan_item["window_id"])
    sequence = str(plan_item["sequence"])
    output = output_root / "windows" / f"{window_id}.jsonl"
    done_path = output_root / "done" / f"{window_id}.json"
    failures: list[str] = []
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.is_file() else {}
    if done.get("status") != "PASS":
        failures.append(f"done_status={done.get('status')}")
    if done.get("written_frame_count") != int(plan_item["frame_end"]) - int(plan_item["frame_start"]) + 1:
        failures.append("done_frame_count_mismatch")
    if not output.is_file():
        failures.append("missing_output_jsonl")
        return {"window_id": window_id, "sequence": sequence, "status": "FAIL", "failures": failures}

    expected = set(range(int(plan_item["frame_start"]), int(plan_item["frame_end"]) + 1))
    seen: set[int] = set()
    candidate_total = 0
    invalid_rows = 0
    invalid_masks = 0
    missing_masks = 0
    degenerate_boxes = 0
    duplicate_native = 0
    mapping_statuses: dict[str, int] = {}
    frame_hash_mismatch = 0
    frame_records: list[dict[str, Any]] = []
    with output.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"json_decode_line_{line_no}:{exc}")
                continue
            frame = int(record.get("frame", -1))
            if frame in seen:
                failures.append(f"duplicate_frame={frame}")
            seen.add(frame)
            if record.get("schema") != "N71_OFFICIAL_SAM3_CANDIDATE_FRAME_V1":
                failures.append(f"schema={record.get('schema')}")
            for key, value in (("sequence", sequence), ("window_id", window_id)):
                if record.get(key) != value:
                    failures.append(f"{key}_mismatch_frame_{frame}")
            if record.get("runtime_future_gt_used") is not False or record.get("runtime_gt_read") is not False:
                failures.append(f"runtime_gt_flag_frame_{frame}")
            if record.get("candidate_set_complete") is not True:
                failures.append(f"candidate_set_incomplete_frame_{frame}")
            # The exporter indexes the sorted image list, not the one-based
            # filename stem.  Verify the same native frame/hash convention.
            image_paths = sorted(
                (DATA_ROOT / "train" / sequence / "img1").glob("*"),
                key=lambda path: int(path.stem),
            )
            image_path = image_paths[frame] if 0 <= frame < len(image_paths) else Path("/nonexistent")
            if image_path.is_file() and str(record.get("frame_hash_sha256")) != digest(image_path):
                frame_hash_mismatch += 1
            rows = record.get("candidates")
            order = record.get("candidate_order")
            if not isinstance(rows, list) or not isinstance(order, list) or len(rows) != int(record.get("candidate_count", -1)):
                failures.append(f"candidate_container_frame_{frame}")
                rows = rows if isinstance(rows, list) else []
            if order != list(range(len(rows))):
                failures.append(f"candidate_order_frame_{frame}")
            natives: set[int] = set()
            for row in rows:
                try:
                    native = int(row["native_tid"])
                    vector = np.asarray(row["machine_embedding"], dtype=np.float64).reshape(-1)
                    box = np.asarray(row["box"], dtype=np.float64).reshape(4)
                    if vector.shape != (512,) or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-6:
                        invalid_rows += 1
                    if not np.all(np.isfinite(box)):
                        invalid_rows += 1
                    elif box[2] <= box[0] or box[3] <= box[1]:
                        # Preserve finite official observations, including
                        # zero-area disappearance rows.  They are not useful
                        # for IoU but must not be silently dropped from the
                        # complete candidate order.
                        degenerate_boxes += 1
                    if int(row["candidate_index"]) not in range(len(rows)):
                        invalid_rows += 1
                    if native in natives:
                        duplicate_native += 1
                    natives.add(native)
                    mapping = row.get("mapping", {})
                    status = str(mapping.get("public_id_status"))
                    mapping_statuses[status] = mapping_statuses.get(status, 0) + 1
                    if mapping.get("public_id") is not None or mapping.get("runtime_future_gt_used") is not False:
                        failures.append(f"fabricated_or_leaky_mapping_frame_{frame}")
                    if status != "EXPLICIT_NEW_BRANCH_PUBLIC_MAPPING_UNAVAILABLE":
                        failures.append(f"unexpected_mapping_status_frame_{frame}")
                    if row.get("mask") is None:
                        missing_masks += 1
                    elif decode_mask(row.get("mask")) is None:
                        invalid_masks += 1
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
            candidate_total += len(rows)
            frame_records.append(record)
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        failures.append(f"missing_frames={missing[:20]}")
    if extra:
        failures.append(f"extra_frames={extra[:20]}")
    if frame_hash_mismatch:
        failures.append(f"frame_hash_mismatch={frame_hash_mismatch}")
    if duplicate_native:
        failures.append(f"duplicate_native={duplicate_native}")
    if invalid_rows:
        failures.append(f"invalid_candidate_rows={invalid_rows}")
    if invalid_masks:
        failures.append(f"invalid_masks={invalid_masks}")
    if missing_masks:
        failures.append(f"missing_masks={missing_masks}")

    event_info = events.get(str(plan_item["event_id"]), {})
    event = event_info.get("event", {})
    gt_id = event.get("dataset_gt_id")
    core_start, core_end = int(plan_item["core_start"]), int(plan_item["core_end"])
    recall_values: list[float] = []
    target_visible = 0
    if gt_id is not None:
        gt_boxes = load_gt_boxes(sequence, int(gt_id))
        for record in frame_records:
            frame = int(record["frame"])
            if not (core_start <= frame <= core_end):
                continue
            gt_box = gt_boxes.get(frame)
            if gt_box is None:
                continue
            target_visible += 1
            recall_values.append(max((box_iou(row.get("box"), gt_box) for row in record.get("candidates", [])), default=0.0))
    recall_ge_05 = sum(v >= 0.5 for v in recall_values)
    return {
        "schema": "N71_CANDIDATE_WINDOW_AUDIT_V1",
        "window_id": window_id,
        "sequence": sequence,
        "event_id": plan_item["event_id"],
        "action_type": plan_item.get("action_type"),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "frame_start": int(plan_item["frame_start"]),
        "frame_end": int(plan_item["frame_end"]),
        "core_start": core_start,
        "core_end": core_end,
        "expected_frame_count": len(expected),
        "observed_frame_count": len(seen),
        "candidate_row_count": candidate_total,
        "candidate_mapping_statuses": mapping_statuses,
        "degenerate_box_count_preserved": degenerate_boxes,
        "missing_mask_count": missing_masks,
        "invalid_mask_count": invalid_masks,
        "target_gt_recall_audit": {
            "source": "posthoc_train_GT_only; not read by exporter",
            "dataset_gt_id": None if gt_id is None else int(gt_id),
            "target_visible_core_frames": target_visible,
            "core_frames_with_max_box_iou_ge_0.5": recall_ge_05,
            "core_target_recall_at_iou_0.5": None if not recall_values else recall_ge_05 / len(recall_values),
            "max_box_iou_median": None if not recall_values else float(np.median(recall_values)),
            "max_box_iou_p90": None if not recall_values else float(np.quantile(recall_values, 0.90)),
        },
        "runtime_future_gt_used": False,
        "not_real_human_evidence": True,
        "output_sha256": digest(output),
        "done_sha256": digest(done_path) if done_path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, default=ROOT / "outputs/n37/real_event_manifest.json")
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    plan_payload = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    item = next(x for x in plan_payload["windows"] if str(x["window_id"]) == args.window_id)
    events = load_event_manifest(args.event_manifest.resolve())
    result = audit_window(item, args.output_root.resolve(), events)
    result["plan_sha256"] = digest(args.plan.resolve())
    result["event_manifest_sha256"] = digest(args.event_manifest.resolve())
    atomic_json(args.audit_output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
