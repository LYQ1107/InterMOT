#!/usr/bin/env python3
"""Join frozen runtime rows with GT only after runtime artifacts are sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.n72r20r3_common import (
    DATASET_ROOT,
    IOU_THRESHOLD,
    R2_ASSET_ROOT,
    ROOT,
    SEQUENCES,
    best_target_candidate,
    gt_by_frame,
    load_sequence,
    read_events,
    read_labels,
    read_zstd_jsonl,
    target_boxes_for_frame,
    target_iou_for_uid,
    write_zstd_jsonl,
)
from scripts.n72r20r3_failure_taxonomy import classify_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=R2_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--presence-dir", type=Path, default=ROOT / "outputs/N72R20R3/presence")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()

    events = read_events(args.asset_root)
    labels = read_labels(args.asset_root)
    runtime_path = args.presence_dir / "frame_runtime_features.jsonl.zst"
    runtime_rows = read_zstd_jsonl(runtime_path)
    oracle_path = args.presence_dir / "oracle_clean_features.jsonl.zst"
    oracle_rows = read_zstd_jsonl(oracle_path) if oracle_path.is_file() else []
    input_rows = runtime_rows + oracle_rows
    frame_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    gt_cache: dict[str, dict[int, list[tuple[int, list[float]]]]] = {}
    label_rows: list[dict[str, Any]] = []
    for row in input_rows:
        sequence = str(row["sequence"])
        frame = int(row["frame"])
        if sequence not in frame_cache:
            frames, _base = load_sequence(args.asset_root, sequence)
            frame_cache[sequence] = {int(payload["frame"]): candidates for payload, candidates in frames}
            target_gt_id = int(labels[sequence]["target_gt_id"])
            gt_cache[sequence] = gt_by_frame(args.dataset_root / "train" / sequence / "gt" / "gt.txt")
        candidates = frame_cache[sequence][frame]
        target_gt_id = int(labels[sequence]["target_gt_id"])
        target_boxes = target_boxes_for_frame(gt_cache[sequence], frame, target_gt_id)
        best_uid, best_iou = best_target_candidate(candidates, target_boxes)
        base_uid = row.get("base_target_candidate_uid")
        learned_uid = row.get("learned_top1_candidate_uid")
        base_iou = target_iou_for_uid(None if base_uid in (None, "", "None") else str(base_uid), candidates, target_boxes)
        learned_iou = target_iou_for_uid(None if learned_uid in (None, "", "None") else str(learned_uid), candidates, target_boxes)
        taxonomy = classify_frame(
            target_gt_present=bool(target_boxes),
            best_iou=best_iou,
            best_uid=best_uid,
            base_uid=None if base_uid in (None, "", "None") else str(base_uid),
            base_iou=base_iou,
        )
        label_rows.append(
            {
                "stage": "N72R20R3",
                "sequence": sequence,
                "frame": frame,
                "state_condition": row["state_condition"],
                "target_gt_id": target_gt_id,
                "target_gt_present": bool(target_boxes),
                "candidate_set_present": bool(target_boxes and best_iou is not None and best_iou >= IOU_THRESHOLD),
                "best_target_candidate_uid": best_uid,
                "best_target_candidate_iou": best_iou,
                "base_candidate_uid": None if base_uid in (None, "", "None") else str(base_uid),
                "base_candidate_target_iou": base_iou,
                "learned_top1_candidate_uid": None if learned_uid in (None, "", "None") else str(learned_uid),
                "learned_top1_target_iou": learned_iou,
                "taxonomy": taxonomy,
                "posthoc_gt_used": True,
                "runtime_future_gt_used": False,
            }
        )
    args.presence_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.presence_dir / "frame_posthoc_labels.jsonl.zst"
    write_zstd_jsonl(output_path, label_rows)
    manifest = {
        "stage": "N72R20R3",
        "status": "PASS_N72R20R3_POSTHOC_LABEL_JOIN",
        "runtime_input": str(runtime_path),
        "runtime_rows": len(runtime_rows),
        "oracle_rows": len(oracle_rows),
        "label_rows": len(label_rows),
        "state_conditions": sorted({str(row["state_condition"]) for row in label_rows}),
        "join_key": ["sequence", "frame", "state_condition"],
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "label_file": str(output_path),
    }
    (args.presence_dir / "posthoc_label_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "runtime_rows": len(runtime_rows), "label_rows": len(label_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
