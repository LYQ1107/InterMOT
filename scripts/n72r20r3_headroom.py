#!/usr/bin/env python3
"""Summarize read-only P2 association headroom for the future R4 stage.

R3 never changes an assignment.  This script joins the runtime feature tape
with the post-hoc taxonomy only after runtime generation and reports the
frames where a target candidate existed but the frozen base assignment was
wrong.  The result is diagnostic evidence for R4, not a new scorer or policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3_common import ROOT, SEQUENCES


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    finite = [number for value in values if (number := _finite(value)) is not None]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _read_rows(presence_dir: Path) -> list[dict[str, Any]]:
    runtime = read_zstd_jsonl(presence_dir / "frame_runtime_features.jsonl.zst")
    labels = read_zstd_jsonl(presence_dir / "frame_posthoc_labels.jsonl.zst")
    label_map = {
        (str(row["sequence"]), int(row["frame"]), str(row["state_condition"])): row
        for row in labels
    }
    rows: list[dict[str, Any]] = []
    for feature in runtime:
        key = (str(feature["sequence"]), int(feature["frame"]), str(feature["state_condition"]))
        label = label_map[key]
        if feature.get("runtime_future_gt_used") is not False or feature.get("posthoc_gt_used") is not False:
            raise ValueError("runtime headroom input has forbidden GT provenance")
        if label.get("posthoc_gt_used") is not True or label.get("runtime_future_gt_used") is not False:
            raise ValueError("headroom taxonomy join is not explicitly post-hoc")
        rows.append({"runtime": feature, "label": label})
    return rows


def _sequence_record(rows: Sequence[dict[str, Any]], sequence: str) -> dict[str, Any]:
    selected = [item for item in rows if str(item["runtime"]["sequence"]) == sequence]
    correct = [
        item
        for item in selected
        if _finite(item["label"].get("learned_top1_target_iou")) is not None
        and float(item["label"]["learned_top1_target_iou"]) >= 0.50
    ]
    return {
        "sequence": sequence,
        "p2_frames": len(selected),
        "learned_top1_correct_count": len(correct),
        "learned_top1_correct_rate": _rate(len(correct), len(selected)),
        "learned_margin": _stats(item["runtime"].get("learned_margin") for item in selected),
        "learned_top1_score": _stats(item["runtime"].get("learned_top1_score") for item in selected),
        "base_assignment_margin": _stats(item["runtime"].get("base_assignment_margin") for item in selected),
        "predicted_motion_iou_of_base_candidate": _stats(item["runtime"].get("predicted_motion_iou_of_base_candidate") for item in selected),
        "candidate_count": _stats(item["runtime"].get("candidate_count") for item in selected),
        "native_continuity_of_base_candidate": _stats(
            1.0 if item["runtime"].get("native_continuity_of_base_candidate") is True else 0.0
            for item in selected
        ),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }


def _examples(rows: Sequence[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda item: (
            -(_finite(item["runtime"].get("learned_margin")) or float("-inf")),
            str(item["runtime"].get("sequence")),
            int(item["runtime"].get("frame", 0)),
        ),
    )
    output = []
    for item in ordered[:limit]:
        runtime = item["runtime"]
        label = item["label"]
        output.append(
            {
                "sequence": runtime["sequence"],
                "frame": int(runtime["frame"]),
                "base_assignment_candidate_uid": runtime.get("base_assignment_candidate_uid"),
                "learned_top1_candidate_uid": runtime.get("learned_top1_candidate_uid"),
                "learned_top1_target_iou": label.get("learned_top1_target_iou"),
                "learned_top1_correct": bool(
                    _finite(label.get("learned_top1_target_iou")) is not None
                    and float(label["learned_top1_target_iou"]) >= 0.50
                ),
                "learned_margin": runtime.get("learned_margin"),
                "base_assignment_margin": runtime.get("base_assignment_margin"),
                "predicted_motion_iou_of_base_candidate": runtime.get("predicted_motion_iou_of_base_candidate"),
                "native_continuity_of_base_candidate": runtime.get("native_continuity_of_base_candidate"),
                "candidate_count": runtime.get("candidate_count"),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": True,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence-dir", type=Path, default=ROOT / "outputs/N72R20R3/presence")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R3/future_association_headroom.json")
    args = parser.parse_args()

    joined = _read_rows(args.presence_dir)
    p2_rows = [
        item
        for item in joined
        if str(item["label"].get("taxonomy")) == "P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"
        and str(item["runtime"].get("state_condition")) == "S1_R2_STYLE_CAUSAL_LEARNED_STATE"
    ]
    decision_boundary_path = ROOT / "outputs/N72R20R2/decision_boundary_audit.json"
    decision_boundary = json.loads(decision_boundary_path.read_text(encoding="utf-8"))
    payload = {
        "stage": "N72R20R3",
        "purpose": "read_only_future_R4_association_headroom",
        "source_stage": "N72R20R2",
        "state_condition": "S1_R2_STYLE_CAUSAL_LEARNED_STATE",
        "selection": "P2_TARGET_AVAILABLE_ASSOCIATION_WRONG only; target candidate exists and frozen base assignment is wrong",
        "p2_frames": len(p2_rows),
        "learned_top1_correct_count": sum(
            _finite(item["label"].get("learned_top1_target_iou")) is not None
            and float(item["label"]["learned_top1_target_iou"]) >= 0.50
            for item in p2_rows
        ),
        "learned_margin": _stats(item["runtime"].get("learned_margin") for item in p2_rows),
        "base_assignment_margin": _stats(item["runtime"].get("base_assignment_margin") for item in p2_rows),
        "predicted_motion_iou_of_base_candidate": _stats(
            item["runtime"].get("predicted_motion_iou_of_base_candidate") for item in p2_rows
        ),
        "candidate_count": _stats(item["runtime"].get("candidate_count") for item in p2_rows),
        "native_continuity_of_base_candidate": _stats(
            1.0 if item["runtime"].get("native_continuity_of_base_candidate") is True else 0.0
            for item in p2_rows
        ),
        "per_sequence": [_sequence_record(p2_rows, sequence) for sequence in SEQUENCES],
        "examples_high_learned_margin": _examples(p2_rows),
        "r2_decision_boundary_reference": {
            "source": str(decision_boundary_path.relative_to(ROOT)),
            "interpretation": decision_boundary.get("interpretation"),
            "c01_rows": decision_boundary.get("c01_rows", []),
            "c01_required_target_column_delta": decision_boundary.get("c01_required_target_column_delta"),
            "runtime_rule_added": False,
            "solver_changed": False,
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "association_rescue": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"p2_frames": len(p2_rows), "output": str(args.output), "status": "PASS_R3_HEADROOM"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
