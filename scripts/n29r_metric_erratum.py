#!/usr/bin/env python3
"""Recompute the original N29-B box metrics without loading SAM3 weights."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.n29_lit_online_replay import _read_gt, _trial_outputs


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "n29" / "n29b_result.json"
DESTINATION = ROOT / "outputs" / "n29r" / "n29b_metric_erratum.json"


def _saved_outputs(rows: list[dict[str, Any]], public_id: int) -> dict[int, list[Any]]:
    outputs: dict[int, list[Any]] = {}
    for row in rows:
        box = row.get("box", row.get("predicted_box"))
        if box is None:
            continue
        frame = int(row["frame"])
        outputs[frame] = [
            SimpleNamespace(
                sam_object_id=int(public_id),
                box_xyxy=np.asarray(box, dtype=float),
            )
        ]
    return outputs


def _box_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_frame = {int(row["frame"]): row.get("box") for row in left}
    right_by_frame = {int(row["frame"]): row.get("box") for row in right}
    frames = sorted(set(left_by_frame) | set(right_by_frame))
    deltas = []
    for frame in frames:
        a, b = left_by_frame.get(frame), right_by_frame.get(frame)
        if a is None or b is None:
            deltas.append(None)
        else:
            deltas.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    finite = [value for value in deltas if value is not None]
    return {
        "frames": frames,
        "max_abs_box_delta": max(finite) if finite else None,
        "elementwise_equal": bool(finite) and all(value == 0.0 for value in finite),
        "per_frame_max_abs_box_delta": deltas,
    }


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    sequences = source.get("sequences", [])
    sequence_results = source.get("sequence_results", [])
    if len(sequences) != 1 or len(sequence_results) != 1:
        raise SystemExit("N29-R erratum expects the original one-sequence N29-B artifact")
    sequence_dir = Path(sequences[0])
    if "val" in sequence_dir.parts or "test" in sequence_dir.parts:
        raise SystemExit(f"blind-boundary refusal: {sequence_dir}")
    item = sequence_results[0]
    gt = _read_gt(sequence_dir)
    dataset_identity = int(item.get("dataset_identity", item["identity"]))
    public_id = int(item["public_id"])
    correction_frame = int(item["correction_frame"])
    end_frame = int(item["end_frame"])
    start_frame = correction_frame + 1
    anchor_rows = item["anchor_future"]["rows"]
    adapted_rows = item["adapted_future"]["rows"]
    anchor = _trial_outputs(
        _saved_outputs(anchor_rows, public_id),
        gt,
        dataset_identity=dataset_identity,
        public_id=public_id,
        start=start_frame,
        end=end_frame,
        require_visible=True,
    )
    adapted = _trial_outputs(
        _saved_outputs(adapted_rows, public_id),
        gt,
        dataset_identity=dataset_identity,
        public_id=public_id,
        start=start_frame,
        end=end_frame,
        require_visible=True,
    )
    result = {
        "protocol": "N29-R1-METRIC-ERRATUM",
        "status": "PASS",
        "val25_read": False,
        "source_artifact": str(SOURCE),
        "sequence": sequence_dir.name,
        "split": item.get("split", "train"),
        "identity_binding": {
            "dataset_identity": dataset_identity,
            "public_id": public_id,
            "sam_object_id": int(item.get("sam_object_id", public_id)),
        },
        "evaluation_window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "gt_visibility_required": True,
        },
        "original_recorded_metrics": {
            "anchor_mean_box_iou": item["anchor_future"].get("mean_box_iou"),
            "adapted_mean_box_iou": item["adapted_future"].get("mean_box_iou"),
            "anchor_error_count_iou_lt_0_5": item["anchor_future"].get("error_count_iou_lt_0_5"),
            "adapted_error_count_iou_lt_0_5": item["adapted_future"].get("error_count_iou_lt_0_5"),
            "recorded_rows_omitted_dataset_identity": True,
        },
        "corrected_anchor": anchor,
        "corrected_adapted": adapted,
        "anchor_adapted_box_comparison": _box_delta(anchor_rows, adapted_rows),
        "bug_root_cause": (
            "The old evaluator passed public_id=100000 to gt[frame].get(...), "
            "although DanceTrack gt.txt keyed the selected target by dataset_identity=0. "
            "Every target was therefore treated as absent and the old mean mixed absent GT with zero IoU."
        ),
        "claim_impact": (
            "The old IoU=0 performance claim is invalid. The corrected anchor is already near-saturated "
            "on this easy four-frame window, and anchor/adapted boxes are identical; this pilot shows no "
            "visible future LoRA gain but cannot establish a general failure."
        ),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
