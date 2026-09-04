#!/usr/bin/env python3
"""Posthoc audit for the frozen N72R7 multi-hypothesis route trigger.

This audit does not alter replay.  It uses the sealed D1/D2 selector traces
and train GT only after runtime artifacts have been written, then measures
whether a target candidate is commonly present in the decoder's top-K while
the greedy top-1 choice is unstable.  The fixed trigger constants are
diagnostic protocol values, not tunable replay thresholds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
TARGET_MANIFEST = ROOT / "outputs/N72R6/recovery_target_stream_manifest_attempt3.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
HORIZONS = (1, 20, 50, 100)
TOP_K = 3
TOP_K_RECALL_TRIGGER = 0.50
TOP1_MISS_GIVEN_TOP_K_TRIGGER = 0.20
IOU_THRESHOLD = 0.50


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
    result: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame, identity = int(parts[0]) - 1, int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            box = [x, y, x + width, y + height]
            if not np.all(np.isfinite(np.asarray(box, dtype=np.float64))):
                raise ValueError(f"non-finite GT box {path}:{line_number}")
            result[frame][identity] = box
    return result


def metric_template() -> dict[str, Any]:
    return {
        "frames": 0,
        "target_visible_frames": 0,
        "pool_target_present_frames": 0,
        "top1_hit_frames": 0,
        "top3_hit_frames": 0,
        "top5_hit_frames": 0,
        "greedy_selected_hit_frames": 0,
        "greedy_selected_frames": 0,
        "top3_present_top1_miss_frames": 0,
        "target_rank_values": [],
        "none_selected_frames": 0,
        "runtime_flag_violations": 0,
    }


def finalize(metric: dict[str, Any]) -> dict[str, Any]:
    values = metric["target_rank_values"]
    result = dict(metric)
    result["target_rank_values"] = list(values)
    for key in ("pool_target_present_frames", "top1_hit_frames", "top3_hit_frames", "top5_hit_frames", "greedy_selected_hit_frames", "greedy_selected_frames", "top3_present_top1_miss_frames", "none_selected_frames"):
        result[key.replace("_frames", "_rate")] = float(metric[key] / metric["target_visible_frames"]) if metric["target_visible_frames"] else None
    result["top1_miss_given_top3_rate"] = float(metric["top3_present_top1_miss_frames"] / metric["top3_hit_frames"]) if metric["top3_hit_frames"] else None
    result["target_rank_median"] = float(np.median(values)) if values else None
    result["target_rank_p90"] = float(np.percentile(values, 90)) if values else None
    return result


def add_frame(metric: dict[str, Any], row: Mapping[str, Any], gt_box: Sequence[float] | None) -> None:
    metric["frames"] += 1
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
        metric["runtime_flag_violations"] += int(row.get(flag) is not False)
    if gt_box is None:
        return
    metric["target_visible_frames"] += 1
    candidates = list(row.get("candidate_rows", []))
    target_uids = {
        str(item["candidate_uid"])
        for item in candidates
        if box_iou(item.get("box_xyxy", []), gt_box) >= IOU_THRESHOLD
    }
    ranked = [str(item["candidate_uid"]) for item in row.get("selection_audit", {}).get("ranked_candidates", [])]
    if len(ranked) != len(candidates) or len(ranked) != len(set(ranked)):
        raise RuntimeError(f"ranked candidate trace is incomplete at frame {row.get('frame')}")
    if target_uids:
        metric["pool_target_present_frames"] += 1
        ranks = [rank + 1 for rank, uid in enumerate(ranked) if uid in target_uids]
        if ranks:
            metric["target_rank_values"].append(min(ranks))
        top1 = bool(set(ranked[:1]) & target_uids)
        top3 = bool(set(ranked[:3]) & target_uids)
        top5 = bool(set(ranked[:5]) & target_uids)
        metric["top1_hit_frames"] += int(top1)
        metric["top3_hit_frames"] += int(top3)
        metric["top5_hit_frames"] += int(top5)
        metric["top3_present_top1_miss_frames"] += int(top3 and not top1)
    selected = row.get("selection_audit", {}).get("selected_candidate_uid")
    metric["none_selected_frames"] += int(selected is None)
    metric["greedy_selected_frames"] += int(selected is not None)
    metric["greedy_selected_hit_frames"] += int(selected is not None and str(selected) in target_uids)


def load_rows(root: Path, event_id: str) -> list[dict[str, Any]]:
    manifest = read_json(root / event_id / "event_manifest.json")
    if manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY":
        raise RuntimeError(f"event is not PASS: {root}/{event_id}")
    path = Path(str(manifest["frames"]))
    if not path.is_absolute():
        path = ROOT / path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 101:
        raise RuntimeError(f"event row count is not 101: {event_id}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-root", required=True)
    parser.add_argument("--d2-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    event_policy = read_json(EVENT_POLICY)
    events = {str(item["event_id"]): dict(item) for item in event_policy.get("events", [])}
    target_manifest = read_json(TARGET_MANIFEST)
    selected = {str(item["event_id"]): dict(item) for item in target_manifest.get("selected", [])}
    if len(selected) != 32 or not set(selected).issubset(events):
        raise RuntimeError("frozen N72R6 event set is not exactly the expected 32-event subset")
    roots = {"D1": Path(args.d1_root), "D2": Path(args.d2_root)}
    for name, root in roots.items():
        if not root.is_absolute():
            roots[name] = ROOT / root
        batch = read_json(roots[name] / "batch_manifest.json")
        if batch.get("status") != "PASS_N72R7_LEARNED_DECODER_BATCH" or int(batch.get("completed_event_count", -1)) != 32:
            raise RuntimeError(f"{name} batch is not complete PASS")
    gt_cache: dict[str, dict[int, dict[int, list[float]]]] = {}
    by_variant: dict[str, dict[str, Any]] = {}
    for variant, root in roots.items():
        aggregate: dict[str, dict[str, Any]] = {str(horizon): metric_template() for horizon in HORIZONS}
        by_action: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: {str(horizon): metric_template() for horizon in HORIZONS})
        by_sequence: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: {str(horizon): metric_template() for horizon in HORIZONS})
        for event_id in sorted(selected):
            event = events[event_id]
            sequence = str(event["sequence"])
            if sequence not in gt_cache:
                gt_cache[sequence] = load_gt(sequence)
            rows = load_rows(root, event_id)
            if int(rows[0].get("frame_horizon", -1)) != 0 or rows[0].get("memory_read") is not False:
                raise RuntimeError(f"event-frame boundary failed: {variant}/{event_id}")
            for horizon in HORIZONS:
                for row in rows[1 : horizon + 1]:
                    target_box = gt_cache[sequence].get(int(row["frame"]), {}).get(int(event["dataset_gt_id"]))
                    add_frame(aggregate[str(horizon)], row, target_box)
                    add_frame(by_action[str(event["action_type"])][str(horizon)], row, target_box)
                    add_frame(by_sequence[sequence][str(horizon)], row, target_box)
        by_variant[variant] = {
            "aggregate": {key: finalize(value) for key, value in aggregate.items()},
            "by_action": {action: {key: finalize(value) for key, value in horizons.items()} for action, horizons in sorted(by_action.items())},
            "by_sequence": {sequence: {key: finalize(value) for key, value in horizons.items()} for sequence, horizons in sorted(by_sequence.items())},
        }
    trigger_evidence = {}
    for variant, values in by_variant.items():
        primary = values["aggregate"]["20"]
        top3_recall = float(primary["top3_hit_frames"] / primary["target_visible_frames"]) if primary["target_visible_frames"] else 0.0
        top1_miss_given_top3 = primary["top1_miss_given_top3_rate"] or 0.0
        trigger_evidence[variant] = {
            "target_visible_frames_h20": primary["target_visible_frames"],
            "top3_recall_h20": top3_recall,
            "top1_miss_given_top3_h20": top1_miss_given_top3,
            "triggered": bool(top3_recall >= TOP_K_RECALL_TRIGGER and top1_miss_given_top3 >= TOP1_MISS_GIVEN_TOP_K_TRIGGER),
        }
    output = {
        "schema_version": "N72R7_TOPK_MULTI_HYPOTHESIS_TRIGGER_AUDIT_V1",
        "status": "PASS_POSTHOC_TOPK_TRIGGER_AUDIT",
        "created_at_utc": now_utc(),
        "source_variants": {key: str(value) for key, value in roots.items()},
        "event_count": len(selected),
        "sequence_count": len({str(item["sequence"]) for item in selected.values()}),
        "top_k": TOP_K,
        "trigger_rule": {
            "top3_recall_h20_at_least": TOP_K_RECALL_TRIGGER,
            "top1_miss_given_top3_h20_at_least": TOP1_MISS_GIVEN_TOP_K_TRIGGER,
            "diagnostic_only": True,
            "future_metrics_used_for_runtime_or_selection": False,
        },
        "trigger_evidence": trigger_evidence,
        "variants": by_variant,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "next_route_if_triggered": "R3_MULTI_HYPOTHESIS_TEMPORAL_SELECTOR",
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    atomic_json(output_path, output)
    print(json.dumps({"status": output["status"], "trigger_evidence": trigger_evidence, "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
