#!/usr/bin/env python3
"""Assemble the N29 audit/evaluation ledger without inventing metrics.

This evaluator distinguishes the bounded official-decoder pilot from a
scientific full-loop result.  It emits correction/video/identity accounting,
keeps measured time as ``null`` when it was not instrumented, and reports
TrackEval metrics as ``NOT_RUN`` until a complete causal MOT stream exists.
It only reads N29 artifacts and enforces the blind-boundary flag.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "n29"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_RUN", "reason": "artifact_missing", "artifact": str(path)}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"status": "NOT_RUN", "reason": "artifact_root_not_object"}


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "val25_read",
        "sequence",
        "split",
        "identity",
        "public_id",
        "prompt_frame",
        "correction_frame",
        "evaluated_future_frames",
        "box_actions",
        "click_count",
        "mask_corrections",
        "anchor_mean_box_iou",
        "adapted_mean_box_iou",
        "future_error_delta",
        "update_status",
        "candidate_bridge_status",
        "measured_seconds",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _pilot_rows(b_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in b_result.get("sequence_results", []):
        anchor = item.get("anchor_future", {})
        adapted = item.get("adapted_future", {})
        rows.append(
            {
                "val25_read": False,
                "sequence": item.get("sequence"),
                "split": item.get("split"),
                "identity": item.get("identity"),
                "public_id": item.get("public_id"),
                "prompt_frame": item.get("prompt_frame"),
                "correction_frame": item.get("correction_frame"),
                "evaluated_future_frames": len(adapted.get("rows", [])),
                "box_actions": item.get("box_actions", 0),
                "click_count": item.get("click_count", 0),
                "mask_corrections": item.get("mask_corrections", 0),
                "anchor_mean_box_iou": anchor.get("mean_box_iou"),
                "adapted_mean_box_iou": adapted.get("mean_box_iou"),
                "future_error_delta": item.get("future_error_delta"),
                "update_status": item.get("update", {}).get("status"),
                "candidate_bridge_status": item.get("candidate_bridge", {}).get("status"),
                "measured_seconds": None,
                "interpretation": "bounded official-decoder mechanism pilot; not an end-to-end MOT result",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    b = _read(args.output_dir / "n29b_result.json")
    a = _read(args.output_dir / "n29a_causal_smoke.json")
    c = _read(args.output_dir / "n29c_result.json")
    d = _read(args.output_dir / "n29d_result.json")
    blind_violation = any(bool(item.get("val25_read", False)) for item in (a, b, c, d))
    rows = _pilot_rows(b)
    trackeval = {
        "status": "NOT_RUN",
        "reason": "no_complete_N29-C_full_loop_output_and_no_mechanism_benefit",
        "HOTA": "NOT_RUN",
        "DetA": "NOT_RUN",
        "AssA": "NOT_RUN",
        "IDF1": "NOT_RUN",
        "IDSW": "NOT_RUN",
        "MOTA": "NOT_RUN",
    }
    result: dict[str, Any] = {
        "protocol": "N29-EVALUATION",
        "status": "PASS" if not blind_violation else "BLOCKED",
        "scientific_result_status": "NOT_RUN_FULL_LOOP_AND_TRACKEVAL",
        "val25_read": False,
        "blind_violation": blind_violation,
        "phase_status": {
            "N29-A": a.get("status", "NOT_RUN"),
            "N29-B": b.get("status", "NOT_RUN"),
            "N29-C": c.get("status", "NOT_RUN"),
            "N29-D": d.get("status", "NOT_RUN"),
            "N29-E": "NOT_RUN",
        },
        "mechanism_pilot": {
            "sequence_count": len(rows),
            "adapter_parameter_count": b.get("adapter_inventory", {}).get("adapter_parameter_count"),
            "adapter_state_count": b.get("adapter_state_count"),
            "rows_are_scientific_full_loop_metrics": False,
        },
        "correction_video_identity_accounting": {
            "correction_events": int(sum(int(row.get("box_actions") or 0) + int(row.get("click_count") or 0) + int(row.get("mask_corrections") or 0) for row in rows)),
            "videos": len({row.get("sequence") for row in rows if row.get("sequence") is not None}),
            "identities": len({(row.get("sequence"), row.get("public_id")) for row in rows if row.get("public_id") is not None}),
            "repeated_correction_events": "NOT_RUN_FULL_LOOP",
            "frames": int(sum(int(row.get("evaluated_future_frames") or 0) for row in rows)),
            "measured_seconds": None,
        },
        "pilot_rows": rows,
        "trackeval": trackeval,
        "not_run_reasons": {
            "N29-C": c.get("reason", "artifact_missing"),
            "N29-D": d.get("reason", "artifact_missing"),
            "TrackEval": trackeval["reason"],
            "confidence_intervals": "no complete multi-sequence full-loop sample",
            "rank8": "not run; rank4 mainline was the frozen pilot",
        },
    }
    _write_json(args.output_dir / "n29_evaluation.json", result)
    _write_rows(args.output_dir / "n29_metrics.csv", rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blind_violation else 2


if __name__ == "__main__":
    raise SystemExit(main())
