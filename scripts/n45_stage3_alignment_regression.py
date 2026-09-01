#!/usr/bin/env python3
"""Targeted regression for N42 dynamic public-ID axis alignment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/n42/replay/runtime/t0/n37-dancetrack0012-0030-recover_identity-297.json"
OUT = ROOT / "outputs/n45/n45_alignment_targeted_regression.json"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    variant = payload["variants"]["M1"]
    no = next(x["candidate_audit"] for x in variant["branches"]["memory_write=False"]["future_trace"] if int(x["frame"]) == 121)
    yes = next(x["candidate_audit"] for x in variant["branches"]["memory_write=True"]["future_trace"] if int(x["frame"]) == 121)
    candidate_key = lambda a: [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in a["candidates"]]
    checks = {"source_runtime_future_gt_false": payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is False, "candidate_rows_equal": candidate_key(no) == candidate_key(yes), "candidate_count_equal": len(no["candidates"]) == len(yes["candidates"]), "public_id_axis_difference_is_allowed_and_must_be_recorded": set(no["public_id_order"]) != set(yes["public_id_order"],), "no_gt_loaded_runtime": no.get("gt_loaded_posthoc") is False and yes.get("gt_loaded_posthoc") is False}
    if not all(checks.values()):
        raise RuntimeError(f"N45 alignment regression failed: {checks}")
    output = {"status": "PASS", "protocol": "N45_PUBLIC_ID_AXIS_ALIGNMENT_REGRESSION_V1", "source": str(SOURCE), "event_id": payload["event_id"], "variant": "M1", "frame": 121, "checks": checks, "public_id_alignment": {"no_write_ids": sorted(set(no["public_id_order"])), "write_baseline_ids": sorted(set(yes["public_id_order"])), "intersection_ids": sorted(set(no["public_id_order"]) & set(yes["public_id_order"]))}, "repair": "align candidate rows by native ID/box/confidence; retain branch-local public-ID axes and record their intersection/differences"}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUT)}))


if __name__ == "__main__":
    main()
