#!/usr/bin/env python3
"""Audit raw target-session availability and the frozen B0 candidate pool.

All runtime sidecars are validated before GT is opened.  GT is used only after
the sealed artifacts pass structural checks, for posthoc candidate-pool oracle
measurements.  No result of this script is consumed by runtime association.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
N72R5 = ROOT / "outputs/N72R5"
N72R5R1 = ROOT / "outputs/N72R5R1"
N72R6 = ROOT / "outputs/N72R6"
OUT = ROOT / "outputs/N72R7/forensic"
EVENT_POLICY = N72R5 / "mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE08 = N72R5R1 / "controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
TARGET_MANIFEST = N72R6 / "recovery_target_stream_manifest_attempt3.json"
REPLAY_ROOT = N72R6 / "public_replay/human_anchor_fallback_attempt1"
PRIVATE_ROOT = N72R5R1 / "simulation_private"
GT_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")
IOU_THRESHOLD = 0.5
HORIZON = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def box_iou(a: Any, b: Any) -> float:
    aa = [float(v) for v in a]
    bb = [float(v) for v in b]
    ix1, iy1 = max(aa[0], bb[0]), max(aa[1], bb[1])
    ix2, iy2 = min(aa[2], bb[2]), min(aa[3], bb[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, aa[2] - aa[0]) * max(0.0, aa[3] - aa[1])
    area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = GT_ROOT / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row: {path}:{line_number}")
            frame = int(parts[0]) - 1
            gt_id = int(parts[1])
            x, y, width, height = (float(item) for item in parts[2:6])
            result[frame][gt_id] = [x, y, x + width, y + height]
    return dict(result)


def candidate_iou(candidates: list[Mapping[str, Any]], gt_box: list[float]) -> tuple[float | None, str | None]:
    best: tuple[float, str] | None = None
    for candidate in candidates:
        box = candidate.get("box_xyxy", candidate.get("box"))
        if box is None:
            continue
        value = box_iou(box, gt_box)
        uid = str(candidate.get("candidate_uid", ""))
        if best is None or value > best[0] or (value == best[0] and uid < best[1]):
            best = (value, uid)
    return (None, None) if best is None else best


def validate_sealed_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    batch = read_json(REPLAY_ROOT / "replay_batch_status.json")
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY" or int(batch.get("completed_event_count", -1)) != 32:
        raise RuntimeError(f"frozen replay is not complete: {batch.get('status')}")
    event_policy = read_json(EVENT_POLICY)
    events = {str(item["event_id"]): item for item in event_policy.get("events", [])}
    stage08 = read_json(STAGE08)
    eligible: set[str] = set()
    for item in stage08.get("events", []):
        branches = {str(row.get("branch")): row for row in item.get("branches", [])}
        if branches.get("B1_SPATIAL_CORRECTION_ONLY", {}).get("action_precondition_status") == "APPLIED":
            eligible.add(str(item["event_id"]))
    if len(eligible) != 32:
        raise RuntimeError(f"expected 32 eligible events, found {len(eligible)}")
    target_manifest = read_json(TARGET_MANIFEST)
    selected = {str(item["event_id"]): item for item in target_manifest.get("selected", [])}
    if len(selected) != 32 or set(selected) != eligible:
        raise RuntimeError("target recovery manifest does not exactly cover eligible events")
    manifests: dict[str, dict[str, Any]] = {}
    for event_id in sorted(eligible):
        path = REPLAY_ROOT / event_id / "event_manifest.json"
        manifest = read_json(path)
        if manifest.get("status") != "PASS_N72R6_C0_C1_EVENT_REPLAY":
            raise RuntimeError(f"event replay is not PASS: {event_id}")
        if event_id not in events:
            raise RuntimeError(f"unexpected event: {event_id}")
        for name in ("c0", "c1"):
            stream = Path(str(manifest[name]["path"]))
            if not stream.is_file() or sha256_file(stream) != str(manifest[name]["sha256"]):
                raise RuntimeError(f"sealed {name} stream hash mismatch: {event_id}")
            rows = read_jsonl(stream)
            if len(rows) != HORIZON + 1:
                raise RuntimeError(f"sealed {name} frame count mismatch: {event_id}")
            expected = list(range(int(manifest["event_frame"]), int(manifest["event_frame"]) + HORIZON + 1))
            if [int(row.get("frame", -1)) for row in rows] != expected:
                raise RuntimeError(f"sealed {name} frame axis mismatch: {event_id}")
            for row in rows:
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
                    if row.get(flag) is not False:
                        raise RuntimeError(f"sealed {name} GT flag violation: {event_id}:{row.get('frame')}:{flag}")
        target_path = Path(str(manifest["target_stream_frames"]))
        if not target_path.is_file():
            raise RuntimeError(f"missing target stream: {event_id}")
        target_rows = read_jsonl(target_path)
        if len(target_rows) != HORIZON + 1:
            raise RuntimeError(f"target stream frame count mismatch: {event_id}")
        for row in target_rows:
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                if row.get(flag) is not False:
                    raise RuntimeError(f"target stream GT/identity flag violation: {event_id}:{row.get('frame')}:{flag}")
        manifests[event_id] = dict(manifest)
    return batch, event_policy, target_manifest, manifests


def main() -> int:
    batch, event_policy, target_manifest, manifests = validate_sealed_inputs()
    events = {str(item["event_id"]): item for item in event_policy["events"]}
    selected = {str(item["event_id"]): item for item in target_manifest["selected"]}
    table: list[dict[str, Any]] = []
    by_event: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    gt_cache: dict[str, dict[int, dict[int, list[float]]]] = {}
    for event_id in sorted(manifests):
        event = events[event_id]
        manifest = manifests[event_id]
        c0_rows = read_jsonl(Path(str(manifest["c0"]["path"])))
        c1_rows = read_jsonl(Path(str(manifest["c1"]["path"])))
        target_rows = read_jsonl(Path(str(manifest["target_stream_frames"])))
        c0_by_frame = {int(row["frame"]): row for row in c0_rows}
        c1_by_frame = {int(row["frame"]): row for row in c1_rows}
        target_by_frame = {int(row["frame"]): row for row in target_rows}
        sequence = str(event["sequence"])
        gt_cache.setdefault(sequence, load_gt(sequence))
        gt = gt_cache[sequence]
        private_path = PRIVATE_ROOT / event_id / "oracle_private_mapping.json"
        private = read_json(private_path)
        target_gt = int(event["dataset_gt_id"])
        target_public = int(manifest["target_public_id"])
        mapping = {int(k): int(v) for k, v in private["dataset_gt_to_public"].items()}
        if mapping.get(target_gt) != target_public:
            raise RuntimeError(f"posthoc mapping mismatch: {event_id}")
        event_counts: Counter[str] = Counter()
        for frame in range(int(manifest["event_frame"]) + 1, int(manifest["event_frame"]) + HORIZON + 1):
            c0 = c0_by_frame[frame]
            c1 = c1_by_frame[frame]
            raw = list(target_by_frame[frame].get("candidate_rows", []))
            if len(raw) > 1:
                raise RuntimeError(f"raw target stream is not singleton: {event_id}:{frame}")
            raw_present = bool(raw)
            raw_candidate = raw[0] if raw else None
            gt_box = gt.get(frame, {}).get(target_gt)
            visible = gt_box is not None
            b0_candidates = list(c0.get("candidate_rows", []))
            c1_target = [item for item in c1.get("candidate_rows", []) if item.get("candidate_kind") == "TARGET_CORRECTION_SESSION_CANDIDATE"]
            c1_uids = {str(item.get("candidate_uid")) for item in c1_target}
            raw_uid = None if raw_candidate is None else str(raw_candidate.get("candidate_uid"))
            raw_retained = bool(raw_uid is not None and raw_uid in c1_uids)
            raw_rejected = bool(raw_present and not raw_retained and c1.get("target_session_future_candidate_raw_present") is True)
            raw_iou = None if not visible or raw_candidate is None else box_iou(raw_candidate.get("box_xyxy"), gt_box)
            b0_iou, b0_uid = (None, None) if not visible else candidate_iou(b0_candidates, gt_box)
            union_iou, union_uid = (None, None) if not visible else candidate_iou(b0_candidates + raw, gt_box)
            if not raw_present:
                primary_class = "A_RAW_TARGET_ROW_MISSING"
            elif raw_rejected:
                primary_class = "D_RAW_TARGET_ROW_REJECTED_BY_HUMAN_GATE"
            elif visible and raw_iou is not None and raw_iou < IOU_THRESHOLD:
                primary_class = "C_RAW_TARGET_ROW_PRESENT_BUT_DRIFTED"
            elif visible and raw_iou is not None and raw_iou >= IOU_THRESHOLD:
                primary_class = "B_RAW_TARGET_ROW_PRESENT_AND_SPATIALLY_CORRECT"
            else:
                primary_class = "RAW_TARGET_PRESENT_GT_NOT_VISIBLE"
            if raw_present:
                event_counts["raw_target_row_present"] += 1
                totals["raw_target_row_present"] += 1
            else:
                event_counts["raw_target_row_missing"] += 1
                totals["raw_target_row_missing"] += 1
            if visible:
                event_counts["target_visible"] += 1
                totals["target_visible"] += 1
                if b0_iou is not None and b0_iou >= IOU_THRESHOLD:
                    event_counts["b0_oracle_hit"] += 1
                    totals["b0_oracle_hit"] += 1
                if raw_iou is not None and raw_iou >= IOU_THRESHOLD:
                    event_counts["target_extra_oracle_hit"] += 1
                    totals["target_extra_oracle_hit"] += 1
                if union_iou is not None and union_iou >= IOU_THRESHOLD:
                    event_counts["union_oracle_hit"] += 1
                    totals["union_oracle_hit"] += 1
            if raw_rejected:
                event_counts["human_gate_rejected"] += 1
                totals["human_gate_rejected"] += 1
            event_counts[primary_class] += 1
            totals[primary_class] += 1
            row = {
                "event_id": event_id,
                "sequence": sequence,
                "action_type": str(event["action_type"]),
                "event_frame": int(manifest["event_frame"]),
                "frame": int(frame),
                "horizon_offset": int(frame - int(manifest["event_frame"])),
                "target_gt_id_posthoc": target_gt,
                "target_public_id": target_public,
                "target_visible_posthoc": visible,
                "classification": primary_class,
                "raw_target_row_present": raw_present,
                "raw_target_candidate_uid": raw_uid,
                "raw_target_box_xyxy": None if raw_candidate is None else raw_candidate.get("box_xyxy"),
                "raw_target_iou_posthoc": raw_iou,
                "raw_target_retained_after_human_gate": raw_retained,
                "raw_target_rejected_by_human_gate": raw_rejected,
                "raw_target_rejection_reason": "HUMAN_ANCHOR_GATE_OR_TARGET_DOMAIN_NONE" if raw_rejected else None,
                "b0_candidate_count": len(b0_candidates),
                "b0_best_iou_posthoc": b0_iou,
                "b0_best_candidate_uid_posthoc": b0_uid,
                "b0_contains_correct_candidate_posthoc": bool(b0_iou is not None and b0_iou >= IOU_THRESHOLD),
                "target_extra_best_iou_posthoc": raw_iou,
                "target_extra_contains_correct_candidate_posthoc": bool(raw_iou is not None and raw_iou >= IOU_THRESHOLD),
                "union_best_iou_posthoc": union_iou,
                "union_best_candidate_uid_posthoc": union_uid,
                "union_contains_correct_candidate_posthoc": bool(union_iou is not None and union_iou >= IOU_THRESHOLD),
                "c1_target_candidate_retained_count": len(c1_target),
                "c1_target_session_raw_present": c1.get("target_session_future_candidate_raw_present"),
                "c1_target_session_accepted_present": c1.get("target_session_future_candidate_present"),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": True,
            }
            table.append(row)
        action_counts[str(event["action_type"])] += 1
        sequence_counts[sequence] += 1
        by_event[event_id] = {
            "event_id": event_id,
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "event_frame": int(manifest["event_frame"]),
            "future_frame_count": HORIZON,
            "counts": dict(sorted(event_counts.items())),
            "b0_recall_over_visible": None if not event_counts["target_visible"] else float(event_counts["b0_oracle_hit"] / event_counts["target_visible"]),
            "target_extra_recall_over_visible": None if not event_counts["target_visible"] else float(event_counts["target_extra_oracle_hit"] / event_counts["target_visible"]),
            "union_recall_over_visible": None if not event_counts["target_visible"] else float(event_counts["union_oracle_hit"] / event_counts["target_visible"]),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        }

    atomic_jsonl(OUT / "candidate_source_table.jsonl", table)
    summary = {
        "schema_version": "N72R7_CANDIDATE_SOURCE_FORENSIC_SUMMARY_V1",
        "status": "PASS_N72R7_CANDIDATE_SOURCE_FORENSICS",
        "event_count": len(by_event),
        "sequence_count": len(sequence_counts),
        "future_frame_count": len(table),
        "action_event_counts": dict(sorted(action_counts.items())),
        "sequence_event_counts": dict(sorted(sequence_counts.items())),
        "classification_counts": dict(sorted((key, int(value)) for key, value in totals.items() if key.startswith(("A_", "B_", "C_", "D_", "RAW_")))),
        "raw_target_row_present_count": int(totals["raw_target_row_present"]),
        "raw_target_row_missing_count": int(totals["raw_target_row_missing"]),
        "human_gate_rejected_count": int(totals["human_gate_rejected"]),
        "target_visible_frame_count": int(totals["target_visible"]),
        "b0_oracle_hit_count": int(totals["b0_oracle_hit"]),
        "target_extra_oracle_hit_count": int(totals["target_extra_oracle_hit"]),
        "union_oracle_hit_count": int(totals["union_oracle_hit"]),
        "b0_pool_recall_over_visible": None if not totals["target_visible"] else float(totals["b0_oracle_hit"] / totals["target_visible"]),
        "target_extra_recall_over_visible": None if not totals["target_visible"] else float(totals["target_extra_oracle_hit"] / totals["target_visible"]),
        "union_pool_recall_over_visible": None if not totals["target_visible"] else float(totals["union_oracle_hit"] / totals["target_visible"]),
        "events": [by_event[key] for key in sorted(by_event)],
        "inputs": {
            "replay_root": str(REPLAY_ROOT),
            "replay_batch_sha256": sha256_file(REPLAY_ROOT / "replay_batch_status.json"),
            "target_manifest": str(TARGET_MANIFEST),
            "target_manifest_sha256": sha256_file(TARGET_MANIFEST),
            "event_policy": str(EVENT_POLICY),
            "event_policy_sha256": sha256_file(EVENT_POLICY),
            "stage08": str(STAGE08),
            "stage08_sha256": sha256_file(STAGE08),
        },
        "iou_threshold": IOU_THRESHOLD,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "candidate_source_summary.json", summary)
    atomic_json(OUT / "stage_02_status.json", {
        "schema_version": "N72R7_STAGE_STATUS_V1",
        "stage": "N72R7-02_CANDIDATE_SOURCE_FORENSICS",
        "status": summary["status"],
        "event_count": summary["event_count"],
        "sequence_count": summary["sequence_count"],
        "future_frame_count": summary["future_frame_count"],
        "classification_counts": summary["classification_counts"],
        "raw_target_row_missing_count": summary["raw_target_row_missing_count"],
        "human_gate_rejected_count": summary["human_gate_rejected_count"],
        "b0_pool_recall_over_visible": summary["b0_pool_recall_over_visible"],
        "union_pool_recall_over_visible": summary["union_pool_recall_over_visible"],
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "next_stage": "N72R7-03_UNION_CANDIDATE_POOL_ORACLE_RECALL",
        "created_at_utc": summary["created_at_utc"],
    })
    print(json.dumps({
        "status": summary["status"],
        "events": summary["event_count"],
        "frames": summary["future_frame_count"],
        "classification_counts": summary["classification_counts"],
        "b0_recall": summary["b0_pool_recall_over_visible"],
        "union_recall": summary["union_pool_recall_over_visible"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
